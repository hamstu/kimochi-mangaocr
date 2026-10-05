/// Upstream whitespace, ellipsis and fullwidth ASCII normalization.
pub fn post_process(text: &str) -> String {
    let chars: Vec<char> = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(|c| if c == '…' { vec!['.'; 3] } else { vec![c] })
        .collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '.' || chars[i] == '・' {
            let mut end = i + 1;
            while end < chars.len() && (chars[end] == '.' || chars[end] == '・') {
                end += 1;
            }
            if end - i >= 2 {
                out.extend(std::iter::repeat_n('．', end - i));
                i = end;
                continue;
            }
        }
        let c = chars[i];
        if i + 1 < chars.len() && matches!(chars[i + 1], 'ﾞ' | 'ﾟ') {
            let base = halfwidth(c);
            let voiced = match (c, chars[i + 1]) {
                ('ｳ', 'ﾞ') => Some('ヴ'),
                ('ｶ'..='ﾄ', 'ﾞ') | ('ﾊ'..='ﾎ', 'ﾞ') => char::from_u32(base as u32 + 1),
                ('ﾊ'..='ﾎ', 'ﾟ') => char::from_u32(base as u32 + 2),
                _ => None,
            };
            if let Some(v) = voiced {
                out.push(v);
                i += 2;
                continue;
            }
        }
        out.push(if c.is_ascii_graphic() {
            char::from_u32(c as u32 + 0xfee0).unwrap()
        } else {
            halfwidth(c)
        });
        i += 1;
    }
    out
}
fn halfwidth(c: char) -> char {
    const KANA: &str = "。「」、・ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン";
    if ('｡'..='ﾝ').contains(&c) {
        KANA.chars().nth(c as usize - 0xff61).unwrap()
    } else {
        c
    }
}
pub(crate) fn decode(ids: &[u32], vocab: &[String]) -> String {
    ids.iter()
        .filter(|&&id| !matches!(id, 0..=4))
        .map(|&id| {
            vocab[id as usize]
                .strip_prefix("##")
                .unwrap_or(&vocab[id as usize])
        })
        .collect::<Vec<_>>()
        .join(" ")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization() {
        assert_eq!(
            post_process(" 日本 語\nABC 123!…・.・"),
            "日本語ＡＢＣ１２３！．．．．．．"
        );
        assert_eq!(post_process("・ ｡ ｶﾞ"), "・。ガ");
        assert_eq!(post_process("ｳﾞﾊﾟﾋﾞﾜﾞ ﾞ ｶﾟ"), "ヴパビワﾞﾞカﾟ");
    }
}
