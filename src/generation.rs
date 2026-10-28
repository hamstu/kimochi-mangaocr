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
pub(crate) fn generate(
    model: &Model,
    encoded: &Tensor,
    device: &Device,
    options: &OcrOptions,
) -> Result<Hypothesis> {
    let mut state = model.start(encoded)?;
    let mut beams = Beams::new(options.beam_width);
    for _ in 0..options.max_new_tokens {
        let ids = Tensor::new(beams.tokens(), device)?.unsqueeze(1)?;
        let scores = model
            .step(&ids, &mut state)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        match beams.advance(&scores, options)? {
            Some(parents) => state.reorder(&Tensor::new(parents, device)?)?,
            None => break,
        }
    }
    beams.best(options)
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
