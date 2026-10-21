//! Text lines grouped into blocks the way mokuro does it.
//!
//! A port of comic-text-detector's `group_output` (utils/textblock.py, the fork mokuro depends
//! on): every DBNet line goes to the YOLO block it overlaps most, a block's direction and font
//! size come from its lines, its lines are put in reading order by their distance from the page's
//! reading origin, a block whose lines fall apart is split there, lines no block took are merged
//! into blocks of their own, and the blocks are ordered on a reading grid. Lines here are the
//! axis-aligned rectangles this crate's line extraction finds, so every rotation angle the
//! reference computes is zero and its vector arithmetic reduces to sums of widths and heights.
//! Where the reference has a quirk that decides its output (the split sorts lines but not their
//! distances), the quirk is kept: the point is to read pages as mokuro reads them.

/// The YOLO head's language class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Eng,
    Ja,
    Unknown,
}

/// `[x1, y1, x2, y2]`, in page pixels.
pub(crate) type Rect = [f64; 4];

/// A block the detector proposed, before lines are given to it.
#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub bbox: Rect,
    pub confidence: f32,
    pub language: Language,
}

#[derive(Debug, Clone)]
pub(crate) struct Block {
    pub bbox: Rect,
    /// In reading order once grouped.
    pub lines: Vec<Rect>,
    pub language: Language,
    pub vertical: bool,
    pub font_size: f64,
    pub confidence: f32,
    distance: Vec<f64>,
    merged: bool,
    /// No line fell in it, so its box stands in for its one line.
    boxed: bool,
}

impl Block {
    fn new(bbox: Rect, lines: Vec<Rect>, language: Language, confidence: f32) -> Self {
        Self {
            bbox,
            lines,
            language,
            vertical: false,
            font_size: -1.,
            confidence,
            distance: Vec::new(),
            merged: false,
            boxed: false,
        }
    }
    fn adjust_bbox(&mut self, with_bbox: bool) {
        let mut b = if with_bbox {
            self.bbox
        } else {
            [f64::MAX, f64::MAX, f64::MIN, f64::MIN]
        };
        for l in &self.lines {
            b = [
                b[0].min(l[0]),
                b[1].min(l[1]),
                b[2].max(l[2]),
                b[3].max(l[3]),
            ];
        }
        self.bbox = b;
    }
}

/// The reference's `union_area`, which is in fact the intersection's area, or -1 with none.
fn union_area(a: Rect, b: Rect) -> f64 {
    let (x1, y1, x2, y2) = (
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    );
    if y2 < y1 || x2 < x1 {
        return -1.;
    }
    (y2 - y1) * (x2 - x1)
}

/// Shapely's `intersects` for two rectangles: touching counts.
fn intersects(a: Rect, b: Rect) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

fn corner_distance(a: Rect, b: Rect) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// `examine_textblk`: direction, font size and each line's distance from the reading origin.
///
/// Vertical when the lines' heights outweigh their widths (twice over for text not classed
/// Japanese). The font size is the lines' mean thickness. Vertical manga reads from the top
/// right, so a column's distance is how far its centre lies from the page's right edge; a
/// horizontal line's, how far below the top.
fn examine(block: &mut Block, im_w: f64, sort: bool) {
    let n = block.lines.len() as f64;
    let height: f64 = block.lines.iter().map(|l| l[3] - l[1]).sum();
    let width: f64 = block.lines.iter().map(|l| l[2] - l[0]).sum();
    let vertical = if block.language == Language::Ja {
        height > width
    } else {
        height > width * 2.
    };
    block.font_size = (if vertical { width } else { height } / n).round();
    block.distance = block
        .lines
        .iter()
        .map(|l| {
            if vertical {
                (im_w - (l[0] + l[2]) / 2.).abs()
            } else {
                ((l[1] + l[3]) / 2.).abs()
            }
        })
        .collect();
    block.vertical = vertical;
    if sort {
        let mut order: Vec<usize> = (0..block.lines.len()).collect();
        order.sort_by(|&a, &b| block.distance[a].total_cmp(&block.distance[b]));
        block.lines = order.iter().map(|&i| block.lines[i]).collect();
        block.distance = order.iter().map(|&i| block.distance[i]).collect();
    }
}

/// `split_textblk`: a block cut where its lines are further apart than two glyphs, or, in a
/// column block, where a column starts more than a glyph above or below the one before it.
fn split(block: &Block) -> (bool, Vec<Block>) {
    let font_size = block.font_size;
    let l0 = block.lines[0];
    let mut lines = block.lines.clone();
    // As the reference does: the lines are re-sorted, their distances are not.
    lines.sort_by(|a, b| corner_distance(*a, l0).total_cmp(&corner_distance(*b, l0)));
    let distance = &block.distance;
    let tolerance = font_size * 2.;
    let mut current = block.clone();
    current.lines = vec![l0];
    let mut subs = vec![current];
    for (jj, &line) in lines.iter().enumerate().skip(1).map(|(i, l)| (i - 1, l)) {
        let previous = lines[jj];
        let mut cut = false;
        if !intersects(previous, line) {
            let apart = (distance[jj + 1] - distance[jj]).abs();
            if apart > tolerance {
                cut = true;
            } else if block.vertical {
                let current = subs.last().expect("one block at least");
                if current.lines.len() > 1 || apart > font_size {
                    cut = (previous[1] - line[1]).abs() > font_size;
                }
            }
        }
        if cut {
            let mut next = subs.last().expect("one block at least").clone();
            next.lines = vec![line];
            subs.push(next);
        } else {
            subs.last_mut()
                .expect("one block at least")
                .lines
                .push(line);
        }
    }
    let splitted = subs.len() > 1;
    if splitted {
        for b in &mut subs {
            b.adjust_bbox(false);
        }
    }
    (splitted, subs)
}

/// `try_merge_textline`: a stray line joins the block before it when they are the same size and
/// close enough in reading order.
fn try_merge(block: &mut Block, other: &mut Block) -> bool {
    if other.merged {
        return false;
    }
    let ratio = block.font_size / other.font_size;
    let (n1, n2) = (block.lines.len() as f64, other.lines.len() as f64);
    let average = (block.font_size * n1 + other.font_size * n2) / (n1 + n2);
    let last = *block.lines.last().expect("a block has lines");
    let other_last = *other.lines.last().expect("a block has lines");
    let distance =
        other.distance.last().copied().unwrap_or(0.) - block.distance.last().copied().unwrap_or(0.);
    let corners = corner_distance(other_last, last);
    if !intersects(last, other_last) {
        if ratio > 1.3 || 1. / ratio > 1.3 {
            return false;
        }
        // Lines of one direction are parallel here, so the reference's 30° test always passes.
        if distance > 2. * average || corners > average * 2.5 {
            return false;
        }
    }
    block.lines.push(other.lines[0]);
    block
        .distance
        .push(other.distance.last().copied().unwrap_or(0.));
    block.font_size = average;
    other.merged = true;
    true
}

fn merge_lines(mut blocks: Vec<Block>) -> Vec<Block> {
    if blocks.len() < 2 {
        return blocks;
    }
    blocks.sort_by(|a, b| a.distance[0].total_cmp(&b.distance[0]));
    let mut merged = Vec::new();
    for i in 0..blocks.len() {
        if blocks[i].merged {
            continue;
        }
        let (head, tail) = blocks.split_at_mut(i + 1);
        for other in tail {
            try_merge(&mut head[i], other);
        }
        merged.push(head[i].clone());
    }
    for b in &mut merged {
        b.adjust_bbox(false);
    }
    merged
}

/// `sort_textblk_list`: blocks in reading order on a grid of three columns and four rows, read
/// right to left when most blocks are Japanese; a spread (wider than tall) reads as two pages.
fn sort_on_grid(blocks: &mut [Block], im_w: f64, im_h: f64) {
    if blocks.is_empty() {
        return;
    }
    let ja = blocks.iter().filter(|b| b.language == Language::Ja).count();
    let flip = ja as f64 > blocks.len() as f64 / 2.;
    let full_width = im_w;
    let page_width = if im_w > im_h { im_w / 2. } else { im_w };
    let (grid_y, grid_x) = (4., 3.);
    let area = im_h * page_width;
    let weight = |b: &Block| {
        let mut cx = (b.bbox[0] + b.bbox[2]) / 2.;
        if flip {
            cx = if page_width != full_width {
                full_width - cx
            } else {
                page_width - cx
            };
        }
        let gx = (cx / page_width * grid_x) as i32 as f64;
        let cy = (b.bbox[1] + b.bbox[3]) / 2.;
        let gy = (cy / im_h * grid_y) as i32 as f64;
        let index = gy * grid_x + gx;
        let mut w =
            index * area + 1.2 * (cx - gx * page_width / grid_x) + (cy - gy * im_h / grid_y);
        if page_width != full_width && gx >= grid_x {
            w += area * grid_y * grid_x;
        }
        w
    };
    blocks.sort_by(|a, b| weight(a).total_cmp(&weight(b)));
}

/// Not in the reference: the parts of a block's box its lines leave out, read as lines too.
///
/// mokuro reads whole pages, where DBNet finds every line. A snip of one bubble gives it less to
/// go on, and it can find one column of three, or a column without its first glyph, while the
/// block's box still holds them all; read line by line, the rest would be lost. So in a block
/// that kept its box (one the reference did not split), a strip across it that no line covers
/// and that is at least two thirds of a glyph wide is read as a line the length of the box, and
/// when every line starts more than a glyph after the box does, the lines that start first are
/// taken back to the box's start. On a full page the box hugs its lines, and where it does not it
/// reaches over a neighbour's text: strips are read only while the lines found cover no more than
/// half the box across, and never where a line was found that is not this block's (`page`).
fn cover(block: &mut Block, page: &[Rect]) {
    if block.lines.is_empty() || block.font_size <= 0. {
        return;
    }
    let fs = block.font_size;
    let v = block.vertical;
    // Across: x for columns, y for rows. Along: y for columns, x for rows.
    let (a0, a1, l0, l1) = if v { (0, 2, 1, 3) } else { (1, 3, 0, 2) };
    let b = block.bbox;
    let first = block.lines.iter().map(|l| l[l0]).fold(f64::MAX, f64::min);
    if first - b[l0] > fs {
        for l in &mut block.lines {
            if l[l0] - first < fs / 2. {
                l[l0] = b[l0];
            }
        }
    }
    let mut covered: Vec<(f64, f64)> = block.lines.iter().map(|l| (l[a0], l[a1])).collect();
    covered.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut edge = b[a0];
    let mut missed = Vec::new();
    for (start, end) in covered.into_iter().chain([(b[a1], b[a1])]) {
        if start - edge >= fs * 2. / 3. {
            let mut l = [0.; 4];
            (l[a0], l[a1], l[l0], l[l1]) = (edge, start, b[l0], b[l1]);
            missed.push(l);
        }
        edge = edge.max(end);
    }
    let width = |l: &Rect| l[a1] - l[a0];
    let missing: f64 = missed.iter().map(width).sum();
    if missed.is_empty() || missing < b[a1] - b[a0] - missing {
        return;
    }
    let own = |l: &Rect| block.lines.iter().any(|o| o == l);
    missed.retain(|m| {
        !page
            .iter()
            .any(|l| !own(l) && union_area(*m, *l) > 0.5 * (l[2] - l[0]) * (l[3] - l[1]))
    });
    block.lines.extend(missed);
    // Reading order again: columns right to left, rows top to bottom.
    if v {
        block
            .lines
            .sort_by(|x, y| (y[0] + y[2]).total_cmp(&(x[0] + x[2])));
    } else {
        block
            .lines
            .sort_by(|x, y| (x[1] + x[3]).total_cmp(&(y[1] + y[3])));
    }
}

/// Not in the reference: a block with no line of its own takes the page's direction.
///
/// The reference judges such a block by its box alone, so a squarish bubble of two short
/// columns reads as one horizontal line. The blocks whose lines say which way they run are
/// a better witness: when most of them agree, the boxed blocks follow them.
fn follow_the_page(blocks: &mut [Block]) {
    let decided = blocks.iter().filter(|b| !b.boxed);
    let vertical = decided.clone().filter(|b| b.vertical).count();
    let horizontal = decided.count() - vertical;
    if vertical == horizontal {
        return;
    }
    for b in blocks.iter_mut().filter(|b| b.boxed) {
        // Its one line is its box, so its font size is the box's thickness that way.
        b.vertical = vertical > horizontal;
        b.font_size = if b.vertical {
            b.bbox[2] - b.bbox[0]
        } else {
            b.bbox[3] - b.bbox[1]
        };
    }
}

/// `group_output`: lines given to blocks, blocks examined, split, merged and ordered.
///
/// `mask` is the mean of the detector's text mask (0 to 1) over a rectangle: a line no block
/// took, or a block no line fell in, is kept only where the mask says there is text.
pub(crate) fn group(
    candidates: &[Candidate],
    lines: &[Rect],
    im_w: f64,
    im_h: f64,
    mask: impl Fn(Rect) -> f64,
) -> Vec<Block> {
    const BBOX_SCORE: f64 = 0.4;
    const MASK_SCORE: f64 = 0.1;
    let mut blocks: Vec<Block> = candidates
        .iter()
        .map(|c| Block::new(c.bbox, Vec::new(), c.language, c.confidence))
        .collect();
    let (mut vertical, mut horizontal) = (Vec::new(), Vec::new());
    for &line in lines {
        let area = (line[3] - line[1]) * (line[2] - line[0]);
        let mut best = (-1., None);
        for (j, b) in blocks.iter().enumerate() {
            let score = union_area(b.bbox, line) / area;
            if best.0 < score {
                best = (score, Some(j));
            }
        }
        match best {
            (score, Some(j)) if score > BBOX_SCORE => blocks[j].lines.push(line),
            _ => {
                if mask(line) < MASK_SCORE {
                    continue;
                }
                let mut b = Block::new(line, vec![line], Language::Unknown, 0.);
                examine(&mut b, im_w, false);
                if b.vertical {
                    vertical.push(b);
                } else {
                    horizontal.push(b);
                }
            }
        }
    }
    let mut out = Vec::new();
    for mut b in blocks {
        if b.lines.is_empty() {
            if mask(b.bbox) < MASK_SCORE {
                continue;
            }
            b.lines = vec![b.bbox];
            b.boxed = true;
        }
        examine(&mut b, im_w, true);
        let (splitted, mut subs) =
            if b.lines.len() > 1 && (b.language == Language::Ja || b.vertical) {
                split(&b)
            } else {
                (false, vec![b])
            };
        if !splitted {
            for s in &mut subs {
                s.adjust_bbox(true);
                cover(s, lines);
            }
        }
        out.extend(subs);
    }
    out.extend(merge_lines(horizontal));
    out.extend(merge_lines(vertical));
    follow_the_page(&mut out);
    sort_on_grid(&mut out, im_w, im_h);
    for b in &mut out {
        if b.language == Language::Eng && !b.vertical && !b.lines.is_empty() {
            let expand = ((b.font_size * 0.1) as i64).max(2) as f64;
            for l in &mut b.lines {
                l[1] = (l[1] - expand).clamp(0., im_h - 1.).trunc();
                l[3] = (l[3] + expand).clamp(0., im_h - 1.).trunc();
            }
            b.font_size += expand;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ja(bbox: Rect) -> Candidate {
        Candidate {
            bbox,
            confidence: 0.9,
            language: Language::Ja,
        }
    }

    #[test]
    fn columns_read_right_to_left() {
        let lines = [
            [10., 0., 30., 100.],
            [70., 0., 90., 100.],
            [40., 0., 60., 80.],
        ];
        let blocks = group(&[ja([0., 0., 100., 100.])], &lines, 800., 1000., |_| 1.);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].vertical);
        assert_eq!(blocks[0].font_size, 20.);
        let xs: Vec<_> = blocks[0].lines.iter().map(|l| l[0]).collect();
        assert_eq!(xs, [70., 40., 10.]);
    }

    #[test]
    fn a_column_starting_well_below_its_neighbour_begins_a_new_block() {
        // Two bubbles the detector boxed as one: the left pair of columns starts 200px lower.
        let lines = [
            [100., 0., 120., 100.],
            [70., 0., 90., 100.],
            [40., 200., 60., 300.],
            [10., 200., 30., 300.],
        ];
        let blocks = group(&[ja([0., 0., 130., 300.])], &lines, 800., 1000., |_| 1.);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].lines.len() + blocks[1].lines.len(), 4);
        for b in &blocks {
            assert_eq!(b.lines.len(), 2);
            assert!(b.bbox[3] - b.bbox[1] <= 100.);
        }
    }

    #[test]
    fn short_text_is_vertical_by_its_lines() {
        // 「……／かう」: the dots' column is squarish, かう's tall.
        let lines = [[163., 130., 186., 159.], [73., 133., 107., 191.]];
        let blocks = group(&[ja([73., 103., 186., 191.])], &lines, 800., 1000., |_| 1.);
        assert!(blocks.iter().all(|b| b.vertical));
    }

    #[test]
    fn a_snip_keeps_the_columns_and_glyphs_its_lines_missed() {
        // One column found of three, and that one without its first glyph.
        let lines = [[140., 60., 170., 200.]];
        let blocks = group(&[ja([20., 0., 170., 200.])], &lines, 200., 210., |_| 1.);
        assert_eq!(blocks.len(), 1);
        let lines = &blocks[0].lines;
        assert_eq!(lines[0], [140., 0., 170., 200.]);
        assert_eq!(lines[1], [20., 0., 140., 200.]);
    }

    #[test]
    fn a_block_without_lines_reads_the_way_the_page_does() {
        let column = [700., 100., 720., 200.];
        let squarish = [300., 300., 370., 360.];
        let blocks = group(
            &[ja([690., 90., 730., 210.]), ja(squarish)],
            &[column],
            800.,
            1000.,
            |_| 1.,
        );
        assert_eq!(blocks.len(), 2);
        assert!(blocks.iter().all(|b| b.vertical));
    }

    #[test]
    fn stray_lines_need_the_mask() {
        let lines = [[500., 500., 520., 600.]];
        assert!(group(&[], &lines, 800., 1000., |_| 0.).is_empty());
        let kept = group(&[], &lines, 800., 1000., |_| 0.5);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].language, Language::Unknown);
        assert!(kept[0].vertical);
    }

    #[test]
    fn stray_neighbouring_columns_merge() {
        let lines = [[500., 500., 520., 600.], [470., 500., 490., 600.]];
        let blocks = group(&[], &lines, 800., 1000., |_| 0.5);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].lines.len(), 2);
        assert_eq!(blocks[0].lines[0][0], 500.);
    }

    #[test]
    fn blocks_follow_the_reading_grid() {
        let left = [100., 100., 120., 200.];
        let right = [700., 100., 720., 200.];
        let lower = [700., 800., 720., 900.];
        let blocks = group(
            &[ja(left), ja(lower), ja(right)],
            &[left, lower, right],
            800.,
            1000.,
            |_| 1.,
        );
        let xs: Vec<_> = blocks.iter().map(|b| (b.bbox[0], b.bbox[1])).collect();
        assert_eq!(xs, [(700., 100.), (100., 100.), (700., 800.)]);
    }
}
