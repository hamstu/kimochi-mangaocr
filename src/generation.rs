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
/// Every token `blocked` would refuse after `ids`.
fn banned(ids: &[u32], n: usize) -> Vec<u32> {
    if n == 0 || ids.len() + 1 < n {
        return Vec::new();
    }
    let prefix = &ids[ids.len() - (n - 1)..];
    ids.windows(n)
        .filter(|w| &w[..n - 1] == prefix)
        .map(|w| w[n - 1])
        .collect()
}
#[cfg(test)]
fn blocked(ids: &[u32], token: u32, n: usize) -> bool {
    if n == 0 || ids.len() + 1 < n {
        return false;
    }
    let prefix = &ids[ids.len() - (n - 1)..];
    ids.windows(n)
        .any(|w| &w[..n - 1] == prefix && w[n - 1] == token)
}
/// One image's beam search, advanced a step at a time from the decoder's log-probabilities,
/// so a decoder can step many images' beams in one batch.
pub(crate) struct Beams {
    live: Vec<Hypothesis>,
    finished: Vec<Hypothesis>,
}
impl Beams {
    pub fn new(width: usize) -> Self {
        let mut live = vec![
            Hypothesis {
                ids: vec![2],
                score: f64::NEG_INFINITY,
                ended: false
            };
            width
        ];
        live[0].score = 0.;
        Self {
            live,
            finished: Vec::new(),
        }
    }
    /// The token each beam feeds the decoder next.
    pub fn tokens(&self) -> Vec<u32> {
        self.live.iter().map(|h| *h.ids.last().unwrap()).collect()
    }
    /// One step from `scores`, a row of vocabulary log-probabilities per beam. Answers the
    /// beam each next beam continues (the decoder's cache order), or `None` once the search
    /// is over.
    pub fn advance(&mut self, scores: &[f32], options: &OcrOptions) -> Result<Option<Vec<u32>>> {
        let b = options.beam_width;
        let mut candidates = Vec::with_capacity(b * 6144);
        for (parent, row) in scores.chunks_exact(6144).enumerate() {
            if !self.live[parent].score.is_finite() {
                continue;
            }
            // The few tokens that would repeat an n-gram, found once per beam (`blocked`'s
            // test, without scanning the history for every token).
            let banned = banned(&self.live[parent].ids, options.no_repeat_ngram_size);
            for (token, &score) in row.iter().enumerate() {
                if score.is_nan() || score == f32::INFINITY {
                    return Err(Error::ModelContract("non-finite decoder logits".into()));
                }
                if banned.contains(&(token as u32)) {
                    continue;
                }
                let value = self.live[parent].score + f64::from(score);
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
            let mut h = self.live[parent].clone();
            h.ids.push(token);
            h.score = score;
            if token == 3 {
                if rank < b {
                    h.ended = true;
                    self.finished.push(h);
                }
            } else {
                parents.push(parent as u32);
                next.push(h);
                if next.len() == b {
                    break;
                }
            }
        }
        self.finished.sort_by(|a, b| {
            b.rank(options.length_penalty)
                .total_cmp(&a.rank(options.length_penalty))
        });
        self.finished.truncate(b);
        if self.finished.len() == b {
            return Ok(None);
        }
        if next.len() != b {
            return Err(Error::ModelContract(
                "decoder produced too few finite candidates".into(),
            ));
        }
        self.live = next;
        Ok(Some(parents))
    }
    /// The best hypothesis once the search is over or the token budget is spent.
    pub fn best(mut self, options: &OcrOptions) -> Result<Hypothesis> {
        // Upstream early_stopping=true excludes live hypotheses once all beams finish.
        if self.finished.len() < options.beam_width {
            self.finished.extend(self.live);
        }
        self.finished
            .into_iter()
            .max_by(|a, b| {
                a.rank(options.length_penalty)
                    .total_cmp(&b.rank(options.length_penalty))
            })
            .ok_or_else(|| Error::ModelContract("empty generation".into()))
    }
}
/// Every image's search at once, their beams stepping through the decoder together, for
/// `encoded`, `[n, 197, 768]`. Images leave the batch a few at a time as they finish (see
/// `onnx::OnnxReader::decode`, which steps the same way).
pub(crate) fn generate(
    model: &Model,
    encoded: &Tensor,
    device: &Device,
    options: &OcrOptions,
) -> Result<Vec<Hypothesis>> {
    use rayon::prelude::*;
    use std::collections::HashMap;
    let w = options.beam_width;
    let images = encoded.dim(0)?;
    let mut state = model.start(encoded, w)?;
    let mut beams: Vec<_> = (0..images).map(|_| Beams::new(w)).collect();
    let mut done = vec![false; images];
    // The images in the batch, in row order: image `active[i]` owns rows `i·w..(i+1)·w`.
    let mut active: Vec<usize> = (0..images).collect();
    for _ in 0..options.max_new_tokens {
        let mut ids = Vec::with_capacity(active.len() * w);
        for &j in &active {
            if done[j] {
                ids.extend(std::iter::repeat_n(3u32, w));
            } else {
                ids.extend(beams[j].tokens());
            }
        }
        let ids = Tensor::new(ids, device)?.unsqueeze(1)?;
        let scores = model
            .step(&ids, &mut state)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let width = w * 6144;
        let row_of: HashMap<usize, usize> = active
            .iter()
            .enumerate()
            .filter(|&(_, &j)| !done[j])
            .map(|(i, &j)| (j, i))
            .collect();
        let advanced: Vec<(usize, Option<Vec<u32>>)> = beams
            .par_iter_mut()
            .enumerate()
            .filter_map(|(j, b)| row_of.get(&j).map(|&i| (j, i, b)))
            .map(|(j, i, b)| Ok((j, b.advance(&scores[i * width..(i + 1) * width], options)?)))
            .collect::<Result<_>>()?;
        let mut parents: HashMap<usize, Vec<u32>> = HashMap::new();
        for (j, p) in advanced {
            match p {
                Some(p) => {
                    parents.insert(j, p);
                }
                None => done[j] = true,
            }
        }
        if done.iter().all(|&d| d) {
            break;
        }
        // Finished images leave once a quarter of the batch has finished.
        let live = active.iter().filter(|&&j| !done[j]).count();
        let keep: Vec<usize> = if live * 4 <= active.len() * 3 {
            active.iter().copied().filter(|&j| !done[j]).collect()
        } else {
            active.clone()
        };
        let mut next = Vec::with_capacity(keep.len() * w);
        let mut kept_rows = Vec::with_capacity(keep.len());
        for &j in &keep {
            let row = active
                .iter()
                .position(|&a| a == j)
                .expect("kept image is active");
            kept_rows.push(row as u32);
            let base = (row * w) as u32;
            match parents.get(&j) {
                Some(p) => next.extend(p.iter().map(|&p| base + p)),
                // A finished image not yet dropped keeps its rows, stepping on EOS.
                None => next.extend((0..w as u32).map(|k| base + k)),
            }
        }
        state.reorder(&Tensor::new(next, device)?)?;
        if keep.len() != active.len() {
            state.keep_images(&Tensor::new(kept_rows, device)?)?;
            active = keep;
        }
    }
    beams.into_iter().map(|b| b.best(options)).collect()
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
    #[test]
    fn banned_tokens_are_exactly_the_blocked_ones() {
        let histories: [&[u32]; 5] = [
            &[2, 10, 11, 12, 10, 11],
            &[2, 10],
            &[2, 7, 7, 7, 7],
            &[2, 5, 6, 5, 6, 5],
            &[2],
        ];
        for ids in histories {
            for n in 0..5 {
                let banned = banned(ids, n);
                for token in 0..20 {
                    assert_eq!(
                        banned.contains(&token),
                        blocked(ids, token, n),
                        "{ids:?} n={n} t={token}"
                    );
                }
            }
        }
    }
}
