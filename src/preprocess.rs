use crate::{Error, Result};
use candle_core::{Device, Tensor};
use image::RgbImage;
const SIZE: usize = 224;
fn weights(from: usize) -> Vec<Vec<(usize, i64)>> {
    let scale = from as f64 / SIZE as f64;
    let support = scale.max(1.);
    (0..SIZE)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(from);
            let mut row: Vec<_> = (start..end)
                .map(|j| {
                    (
                        j,
                        (1. - ((j as f64 + 0.5 - center) / support).abs()).max(0.),
                    )
                })
                .collect();
            let sum: f64 = row.iter().map(|(_, w)| w).sum();
            for (_, w) in &mut row {
                *w /= sum;
            }
            row.into_iter()
                .map(|(j, w)| (j, (w * 4194304.).round() as i64))
                .collect()
        })
        .collect()
}
/// Pillow-compatible grayscale followed by antialiased bilinear square resize.
pub(crate) fn pixels(image: &RgbImage, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        pixel_data(image)?,
        (1, 3, SIZE, SIZE),
        device,
    )?)
}
/// `pixels` as plain NCHW data, for a runtime other than Candle.
pub(crate) fn pixel_data(image: &RgbImage) -> Result<Vec<f32>> {
    validate(image)?;
    let (w, h) = (image.width() as usize, image.height() as usize);
    let gray: Vec<u8> = image
        .pixels()
        .map(|p| {
            ((19595 * u32::from(p[0]) + 38470 * u32::from(p[1]) + 7471 * u32::from(p[2]) + 32768)
                >> 16) as u8
        })
        .collect();
    let wx = weights(w);
    let wy = weights(h);
    let mut horizontal = vec![0u8; SIZE * h];
    for y in 0..h {
        for (x, row) in wx.iter().enumerate() {
            let sum: i64 = row
                .iter()
                .map(|&(j, a)| i64::from(gray[y * w + j]) * a)
                .sum();
            horizontal[y * SIZE + x] = ((sum + 2097152) >> 22).clamp(0, 255) as u8;
        }
    }
    let mut out = vec![0f32; 3 * SIZE * SIZE];
    for (y, row) in wy.iter().enumerate() {
        for x in 0..SIZE {
            let sum: i64 = row
                .iter()
                .map(|&(j, a)| i64::from(horizontal[j * SIZE + x]) * a)
                .sum();
            let value = ((sum + 2097152) >> 22).clamp(0, 255) as f32 / 127.5 - 1.;
            for c in 0..3 {
                out[c * SIZE * SIZE + y * SIZE + x] = value;
            }
        }
    }
    Ok(out)
}
pub(crate) fn validate(image: &RgbImage) -> Result<()> {
    if image.width() == 0 || image.height() == 0 {
        return Err(Error::InvalidInput("image must be nonempty".into()));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grayscale_and_channels() {
        let im = RgbImage::from_pixel(7, 13, image::Rgb([255, 0, 0]));
        let v = pixels(&im, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(v.len(), 3 * 224 * 224);
        assert!(v.iter().all(|x| (*x - (76. / 127.5 - 1.)).abs() < 1e-6));
    }
    #[test]
    fn pillow_resize_reference() {
        use sha2::{Digest, Sha256};
        // Pillow 12.3.0 RGB -> L -> RGB -> BILINEAR; hash of the first channel.
        for (w, h, expected) in [
            (
                13,
                7,
                "52cb13c30c2f02f7bac7fe01c4b54de43bc381ba36e566bfb6ec26117706dcb0",
            ),
            (
                513,
                377,
                "e67dd6566c2abafe1322968b379942df3932fde0f735687add9ec02c35e36520",
            ),
            (
                224,
                224,
                "74d03e8f9e2c4c6c9caf3e910cafdfe93ce041396a70002cb8600e1f7b266ec3",
            ),
        ] {
            let im = RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([
                    ((x * 17 + y * 11) % 256) as u8,
                    ((x * 7 + y * 29) % 256) as u8,
                    ((x * 31 + y * 3) % 256) as u8,
                ])
            });
            let v = pixels(&im, &Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let bytes: Vec<_> = v[..224 * 224]
                .iter()
                .map(|x| ((x + 1.) * 127.5).round() as u8)
                .collect();
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected, "{w}x{h}");
        }
    }
    #[test]
    fn rejects_empty() {
        assert!(pixels(&RgbImage::new(0, 1), &Device::Cpu).is_err());
    }
}
