use rust_i18n::t;
use strum_macros::{EnumIter, EnumString};

#[derive(Copy, Clone, Debug, EnumIter, EnumString, PartialEq, Eq)]
pub enum KokoroVoice {
    #[strum(serialize = "Women1")]
    Women1 = 0,
    #[strum(serialize = "Women2")]
    Women2 = 1,
    #[strum(serialize = "Man1")]
    Man1 = 2,
    #[strum(serialize = "Man2")]
    Man2 = 3,
}

impl KokoroVoice {
    pub fn default() -> Self {
        KokoroVoice::Women1
    }
    //获取voice的名称
    pub fn get_name(&self) -> String {
        match self {
            Self::Women1 => t!("women1").as_ref().to_string(),
            Self::Women2 => t!("women2").as_ref().to_string(),
            Self::Man1 => t!("man1").as_ref().to_string(),
            Self::Man2 => t!("man2").as_ref().to_string(),
        }
    }
    //获取voice是否为男性，是则返回true
    pub fn get_is_man(&self) -> bool {
        match self {
            Self::Women1 => false,
            Self::Women2 => false,
            Self::Man1 => true,
            Self::Man2 => true,
        }
    }
    // 从字符串转换的便捷方法
    pub fn from_string(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}
impl From<KokoroVoice> for i32 {
    fn from(voice: KokoroVoice) -> i32 {
        voice as i32
    }
}
