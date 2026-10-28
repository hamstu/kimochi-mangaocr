use crate::model::Model;
use crate::{Error, OcrOptions, Result};
use candle_core::{Device, Tensor};

#[derive(Clone, Debug)]
pub(crate) struct Hypothesis {
    pub ids: Vec<u32>,
    pub score: f64,
    pub ended: bool,
}
impl Hypothesis {
    fn rank(&self, penalty: f64) -> f64 {
        self.score / ((self.ids.len() - 1).max(1) as f64).powf(penalty)
    }
}
fn blocked(ids: &[u32], token: u32, n: usize) -> bool {
    if n == 0 || ids.len() + 1 < n {
        return false;
    }
    let prefix = &ids[ids.len() - (n - 1)..];
    ids.windows(n)
        .any(|w| &w[..n - 1] == prefix && w[n - 1] == token)
}
pub(crate) fn generate(
    model: &Model,
    encoded: &Tensor,
    device: &Device,
    options: &OcrOptions,
) -> Result<Hypothesis> {
    let b = options.beam_width;
    let mut state = model.start(encoded)?;
    let mut live = vec![
        Hypothesis {
            ids: vec![2],
            score: f64::NEG_INFINITY,
            ended: false
        };
        b
    ];
    live[0].score = 0.;
    let mut finished = Vec::new();
    for _ in 0..options.max_new_tokens {
        let ids = Tensor::new(
            live.iter()
                .map(|h| *h.ids.last().unwrap())
                .collect::<Vec<_>>(),
            device,
        )?
        .unsqueeze(1)?;
        let scores = model.step(&ids, &mut state)?.to_vec2::<f32>()?;
        let mut candidates = Vec::with_capacity(b * 6144);
        for (parent, row) in scores.iter().enumerate() {
            if !live[parent].score.is_finite() {
                continue;
            }
            for (token, &score) in row.iter().enumerate() {
                if score.is_nan() || score == f32::INFINITY {
                    return Err(Error::ModelContract("non-finite decoder logits".into()));
                }
                if blocked(
                    &live[parent].ids,
                    token as u32,
                    options.no_repeat_ngram_size,
                ) {
                    continue;
                }
                let value = live[parent].score + f64::from(score);
                if value.is_finite() {
                    candidates.push((value, parent, token as u32));
                }
            }
        }
        let compare = |a: &(f64, usize, u32), b: &(f64, usize, u32)| {
            b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2))
        };
        if candidates.len() > 2 * b {
            candidates.select_nth_unstable_by(2 * b, compare);
            candidates.truncate(2 * b);
        }
        candidates.sort_unstable_by(compare);
        let mut next = Vec::with_capacity(b);
        let mut parents = Vec::with_capacity(b);
        for (rank, (score, parent, token)) in candidates.into_iter().take(2 * b).enumerate() {
            let mut h = live[parent].clone();
            h.ids.push(token);
            h.score = score;
            if token == 3 {
                if rank < b {
                    h.ended = true;
                    finished.push(h);
                }
            } else {
                parents.push(parent as u32);
                next.push(h);
                if next.len() == b {
                    break;
                }
            }
        }
        finished.sort_by(|a, b| {
            b.rank(options.length_penalty)
                .total_cmp(&a.rank(options.length_penalty))
        });
        finished.truncate(b);
        if finished.len() == b {
            break;
        }
        if next.len() != b {
            return Err(Error::ModelContract(
                "decoder produced too few finite candidates".into(),
            ));
        }
        state.reorder(&Tensor::new(parents, device)?)?;
        live = next;
    }
    // Upstream early_stopping=true excludes live hypotheses once all beams finish.
    if finished.len() < b {
        finished.extend(live);
    }
    finished
        .into_iter()
        .max_by(|a, b| {
            a.rank(options.length_penalty)
                .total_cmp(&b.rank(options.length_penalty))
        })
        .ok_or_else(|| Error::ModelContract("empty generation".into()))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ngram_blocking() {
        assert!(blocked(&[2, 10, 11, 12, 10, 11], 12, 3));
        assert!(!blocked(&[2, 10, 11, 12, 10, 11], 13, 3));
        assert!(!blocked(&[2, 10], 10, 0));
        assert!(blocked(&[2, 10], 10, 1));
    }
}
