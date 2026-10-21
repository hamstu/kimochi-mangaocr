//! One text line made into the crops Manga OCR reads, as mokuro makes them.
//!
//! A port of comic-text-detector's `TextBlock.get_transformed_region` and mokuro's
//! `MangaPageOcr.split_into_chunks`: the line is resampled to a thickness of 64 pixels (a column
//! turned on its side, so both directions run left to right), and a line longer than the model
//! reads well (16 glyphs for a column, 8 for a row) is cut into that many pieces, each cut at the
//! emptiest point of the text mask near where an even cut would fall, so no glyph is halved. A
//! column's pieces are turned upright again before they are read.
use crate::blocks::Language;
use crate::detector::TextMask;
use image::{GrayImage, Luma, RgbImage, imageops};

const TEXT_HEIGHT: u32 = 64;
const MAX_RATIO_VERTICAL: f64 = 16.;
const MAX_RATIO_HORIZONTAL: f64 = 8.;
/// How far either side of an even cut a better one is looked for, in line thicknesses.
const ANCHOR_WINDOW: u32 = 2;

/// Bilinear sample at a page position, clamped to the page.
fn sample(image: &RgbImage, x: f64, y: f64) -> [f64; 3] {
    let (w, h) = (image.width() as f64, image.height() as f64);
    let x = x.clamp(0., w - 1.);
    let y = y.clamp(0., h - 1.);
    let (x0, y0) = (x.floor(), y.floor());
    let (x1, y1) = ((x0 + 1.).min(w - 1.), (y0 + 1.).min(h - 1.));
    let (ax, ay) = (x - x0, y - y0);
    let p = |x: f64, y: f64| image.get_pixel(x as u32, y as u32).0;
    let (a, b, c, d) = (p(x0, y0), p(x1, y0), p(x0, y1), p(x1, y1));
    let mut out = [0.; 3];
    for i in 0..3 {
        let top = f64::from(a[i]) * (1. - ax) + f64::from(b[i]) * ax;
        let bottom = f64::from(c[i]) * (1. - ax) + f64::from(d[i]) * ax;
        out[i] = top * (1. - ay) + bottom * ay;
    }
    out
}

/// The crops one line is read as, in reading order.
pub(crate) fn chunks(
    image: &RgbImage,
    mask: Option<&TextMask>,
    rect: [u32; 4],
    vertical: bool,
    language: Language,
    font_size: f32,
) -> Vec<RgbImage> {
    let (im_w, im_h) = (f64::from(image.width()), f64::from(image.height()));
    let mut r = rect.map(f64::from);
    if language == Language::Eng || (language == Language::Unknown && !vertical) {
        let e = f64::from(font_size) / 3.;
        r = [
            (r[0] - e).clamp(0., im_w),
            (r[1] - e).clamp(0., im_h),
            (r[2] + e).clamp(0., im_w),
            (r[3] + e).clamp(0., im_h),
        ];
    }
    let (line_w, line_h) = ((r[2] - r[0]).max(1.), (r[3] - r[1]).max(1.));
    let ratio = line_h / line_w;
    let (w, h) = if vertical {
        (
            TEXT_HEIGHT,
            ((f64::from(TEXT_HEIGHT) * ratio).round() as u32).max(1),
        )
    } else {
        (
            ((f64::from(TEXT_HEIGHT) / ratio).round() as u32).max(1),
            TEXT_HEIGHT,
        )
    };
    // The perspective transform from the line's corners to the crop's, which for an upright
    // rectangle is a scale: corner pixel to corner pixel.
    let at = |u: u32, v: u32| {
        let sx = if w > 1 { line_w / f64::from(w - 1) } else { 0. };
        let sy = if h > 1 { line_h / f64::from(h - 1) } else { 0. };
        (r[0] + f64::from(u) * sx, r[1] + f64::from(v) * sy)
    };
    let crop = RgbImage::from_fn(w, h, |u, v| {
        let (x, y) = at(u, v);
        image::Rgb(sample(image, x, y).map(|c| c.round().clamp(0., 255.) as u8))
    });
    let crop = if vertical {
        imageops::rotate270(&crop)
    } else {
        crop
    };
    let (length, thickness) = (crop.width(), crop.height());
    let max_ratio = if vertical {
        MAX_RATIO_VERTICAL
    } else {
        MAX_RATIO_HORIZONTAL
    };
    let line_ratio = f64::from(length) / f64::from(thickness);
    let upright = |c: RgbImage| if vertical { imageops::rotate90(&c) } else { c };
    if line_ratio <= max_ratio {
        return vec![upright(crop)];
    }
    let pieces = (line_ratio / max_ratio).ceil() as usize;
    // Ink along the line, from the mask resampled as the crop was.
    let density = mask.map(|mask| {
        let plane = GrayImage::from_fn(w, h, |u, v| {
            let (x, y) = at(u, v);
            Luma([(mask.at(x, y) * 255.).round().clamp(0., 255.) as u8])
        });
        let plane = if vertical {
            imageops::rotate270(&plane)
        } else {
            plane
        };
        (0..length)
            .map(|x| {
                (0..thickness)
                    .map(|y| f64::from(plane.get_pixel(x, y)[0]))
                    .sum::<f64>()
            })
            .collect::<Vec<_>>()
    });
    let density = density.map(|d| smoothed(&d, thickness));
    let window = ANCHOR_WINDOW * thickness;
    let mut cuts = Vec::with_capacity(pieces - 1);
    for k in 1..pieces {
        let anchor = (f64::from(length) * k as f64 / pieces as f64) as u32;
        let n0 = anchor.saturating_sub(window / 2).min(length);
        let n1 = (anchor + window / 2).min(length);
        let cut = match &density {
            Some(d) if n1 > n0 => {
                let mut best = n0;
                for p in n0..n1 {
                    if d[p as usize] < d[best as usize] {
                        best = p;
                    }
                }
                best
            }
            _ => anchor,
        };
        cuts.push(cut);
    }
    let mut out = Vec::with_capacity(pieces);
    let mut from = 0;
    for to in cuts.into_iter().chain([length]) {
        if to > from {
            out.push(upright(
                imageops::crop_imm(&crop, from, 0, to - from, thickness).to_image(),
            ));
            from = to;
        }
    }
    out
}

/// `np.convolve(density, gaussian(2t, t / 8), "same")`, normalised to its maximum.
fn smoothed(density: &[f64], thickness: u32) -> Vec<f64> {
    let m = (2 * thickness) as usize;
    let std = f64::from(thickness) / 8.;
    let centre = (m as f64 - 1.) / 2.;
    let kernel: Vec<f64> = (0..m)
        .map(|n| (-0.5 * ((n as f64 - centre) / std).powi(2)).exp())
        .collect();
    let n = density.len();
    let length = n.max(m);
    let offset = (m - 1) / 2;
    let mut out = vec![0.; length];
    for (i, value) in out.iter_mut().enumerate() {
        let full = i + offset;
        let mut sum = 0.;
        for (j, d) in density.iter().enumerate() {
            if full >= j && full - j < m {
                sum += d * kernel[full - j];
            }
        }
        *value = sum;
    }
    let max = out.iter().copied().fold(0., f64::max);
    if max > 0. {
        out.iter_mut().for_each(|v| *v /= max);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_column_is_one_upright_crop_64_wide() {
        let page = RgbImage::from_pixel(200, 400, image::Rgb([255; 3]));
        let crops = chunks(&page, None, [10, 10, 42, 170], true, Language::Ja, 32.);
        assert_eq!(crops.len(), 1);
        assert_eq!(crops[0].width(), 64);
        assert_eq!(crops[0].height(), 320);
    }

    #[test]
    fn a_long_row_is_cut_where_the_mask_is_empty() {
        let page = RgbImage::from_pixel(1000, 100, image::Rgb([255; 3]));
        // A row 20 glyphs long, 10px thick: at 64px thick 1280px, over 8 thicknesses, so three pieces.
        let crops = chunks(&page, None, [0, 0, 200, 10], false, Language::Ja, 10.);
        assert_eq!(crops.len(), 3);
        assert!(crops.iter().all(|c| c.height() == 64));
        assert_eq!(crops.iter().map(|c| c.width()).sum::<u32>(), 1280);
    }

    #[test]
    fn smoothing_matches_numpy_same_mode() {
        let d = vec![0., 0., 1., 0., 0.];
        let s = smoothed(&d, 1);
        // gaussian(2, 1/8) is two equal taps; "same" keeps max(len, M) = 5 values.
        assert_eq!(s.len(), 5);
        assert_eq!(s.iter().filter(|v| **v == 1.).count(), 2);
    }
}
