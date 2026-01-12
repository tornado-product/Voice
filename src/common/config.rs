use crate::common::error::{AgentError, Result};
use log::debug;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::time::Duration;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// 音频样本数据
#[derive(Debug, Clone)]
pub struct AudioData {
    /// 原始音频样本（多通道交错）
    pub samples: Vec<f32>,
    /// 采样率（Hz）
    pub sample_rate: u32,
    /// 通道数
    pub channels: u16,
    /// 持续时间（秒）
    pub duration: f64,
    /// 原始格式
    pub format: AudioFormat,
}
// 音频数据
impl AudioData {
    /// 创建新的音频数据
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16, format: AudioFormat) -> Self {
        let duration = samples.len() as f64 / (sample_rate as f64 * channels as f64);
        Self {
            samples,
            sample_rate,
            channels,
            duration,
            format,
        }
    }

    /// 获取帧数（每个通道的样本数）
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// 通过平均通道转换为单声道
    pub fn to_mono(&self) -> AudioData {
        if self.channels == 1 {
            return self.clone();
        }

        let mut mono_samples = Vec::with_capacity(self.frames());
        for frame in self.samples.chunks(self.channels as usize) {
            let sum: f32 = frame.iter().sum();
            mono_samples.push(sum / self.channels as f32);
        }

        AudioData::new(mono_samples, self.sample_rate, 1, self.format)
    }

    /// 使用线性插值重新采样到目标采样率
    pub fn resample(&self, target_sample_rate: u32) -> Result<AudioData> {
        if self.sample_rate == target_sample_rate {
            return Ok(self.clone());
        }

        let ratio = target_sample_rate as f64 / self.sample_rate as f64;
        let target_frames = (self.frames() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(target_frames * self.channels as usize);

        for target_frame in 0..target_frames {
            let source_frame = target_frame as f64 / ratio;
            let source_frame_floor = source_frame.floor() as usize;
            let source_frame_ceil = (source_frame_floor + 1).min(self.frames() - 1);
            let fraction = source_frame - source_frame_floor as f64;

            for channel in 0..self.channels as usize {
                let sample_floor =
                    self.samples[source_frame_floor * self.channels as usize + channel];
                let sample_ceil =
                    self.samples[source_frame_ceil * self.channels as usize + channel];
                let interpolated = sample_floor + (sample_ceil - sample_floor) * fraction as f32;
                resampled.push(interpolated);
            }
        }

        Ok(AudioData::new(
            resampled,
            target_sample_rate,
            self.channels,
            self.format,
        ))
    }

    // 将音频标准化到目标峰值水平
    /*pub fn normalize(&self, target_peak: f32) -> AudioData {
        let current_peak = self.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if current_peak == 0.0 {
            return self.clone();
        }

        let gain = target_peak / current_peak;
        let normalized_samples: Vec<f32> = self.samples.iter().map(|s| s * gain).collect();

        AudioData::new(
            normalized_samples,
            self.sample_rate,
            self.channels,
            self.format,
        )
    }*/
}

/// 语音转文本服务配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionConfig {
    /// 服务提供商 (whisper, azure, google等)
    pub provider: String,
    /// API的URL
    pub endpoint: Option<String>,
    /// 用于身份验证的API密钥
    pub api_key: Option<String>,
    /// 要使用的模型 (例如"whisper-1", "base", "small"等)
    pub model: String,
    /// 语言代码 (例如"en", "es", "fr")
    pub language: Option<String>,
    /// 是否启用自动语言检测
    pub auto_detect_language: bool,
    /// 请求超时（秒）
    pub timeout: u64,
    /// 失败请求的最大重试次数
    pub max_retries: u32,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            provider: "whisper".to_string(),
            endpoint: None,
            api_key: None,
            model: "base".to_string(),
            language: Some("zh-CN".to_string()),
            auto_detect_language: true,
            timeout: 60,
            max_retries: 3,
        }
    }
}

/// Whisper 状态信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperStatus {
    /// 模型是否已加载
    pub is_loaded: bool,
    /// 是否使用 GPU
    pub is_using_gpu: bool,
    /// 是否使用了 CPU 回退
    pub was_fallback_used: bool,
    /// 后端名称
    pub backend_name: String,
}

/// 音效配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct EffectsConfig {
    /// 启用降噪功能
    pub enable_noise_reduction: bool,
    /// 降噪强度（0.0至1.0）
    pub noise_reduction_strength: f32,
    /// 启用音频标准化
    pub enable_normalization: bool,
    /// 目标归一化水平（dB）
    pub normalization_target: f32,
    /// 启用高通滤波器
    pub enable_highpass_filter: bool,
    /// 高通滤波器截止频率（Hz）
    pub highpass_cutoff: f32,
    /// 启用低通滤波器
    pub enable_lowpass_filter: bool,
    /// 低通滤波器截止频率（Hz）
    pub lowpass_cutoff: f32,
}

impl Default for EffectsConfig {
    fn default() -> Self {
        Self {
            enable_noise_reduction: false,
            noise_reduction_strength: 0.5,
            enable_normalization: false,
            normalization_target: -23.0, // LUFS standard
            enable_highpass_filter: false,
            highpass_cutoff: 80.0,
            enable_lowpass_filter: false,
            lowpass_cutoff: 8000.0,
        }
    }
}

/// 音频格式枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
    Aac,
    M4a,
}

impl AudioFormat {
    /// 获取格式的文件扩展名
    pub fn extension(&self) -> &'static str {
        match self {
            AudioFormat::Wav => "wav",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Flac => "flac",
            AudioFormat::Ogg => "ogg",
            AudioFormat::Aac => "aac",
            AudioFormat::M4a => "m4a",
        }
    }

    /// 从文件扩展名检测格式
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "wav" => Some(AudioFormat::Wav),
            "mp3" => Some(AudioFormat::Mp3),
            "flac" => Some(AudioFormat::Flac),
            "ogg" => Some(AudioFormat::Ogg),
            "aac" => Some(AudioFormat::Aac),
            "m4a" => Some(AudioFormat::M4a),
            _ => None,
        }
    }

    /// 检查格式是否支持元数据
    #[allow(dead_code)]
    pub fn supports_metadata(&self) -> bool {
        matches!(
            self,
            AudioFormat::Mp3 | AudioFormat::Flac | AudioFormat::M4a
        )
    }
}

/// 音频质量设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQuality {
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: u16,
    pub bitrate: Option<u32>, // 对于压缩格式
}

impl Default for AudioQuality {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            bit_depth: 16,
            bitrate: Some(128000), // 128 kbps
        }
    }
}

impl AudioQuality {
    /// 高品质预设（48kHz，24位，立体声）
    #[allow(dead_code)]
    pub fn high() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            bit_depth: 24,
            bitrate: Some(320000), // 320 kbps
        }
    }

    /// 低质量预设（22kHz，16位，单声道）
    #[allow(dead_code)]
    pub fn low() -> Self {
        Self {
            sample_rate: 22050,
            channels: 1,
            bit_depth: 16,
            bitrate: Some(64000), // 64 kbps
        }
    }

    /// 语音质量预设（16kHz，16位，单声道）
    pub fn voice() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            bit_depth: 16,
            bitrate: Some(32000), // 32 kbps
        }
    }
}

/// 音频处理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct AudioConfig {
    /// 最大文件大小（字节）（默认值：100MB）
    pub max_file_size: usize,
    /// 最大处理持续时间（秒）（默认值：300秒=5分钟）
    pub max_processing_duration: u64,
    /// 默认音频质量设置
    pub default_quality: AudioQuality,
    /// 启用已处理音频的缓存
    pub enable_caching: bool,
    /// 缓存已处理音频的TTL（默认值：1小时）
    pub cache_ttl: Duration,
    /// 启用资源监控
    pub enable_monitoring: bool,
    /// stt服务配置
    pub transcription: TranscriptionConfig,
    /// tts服务配置
    pub synthesis: SynthesisConfig,
    /// 音效配置
    pub effects: EffectsConfig,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_processing_duration: 300,     // 5 分钟
            default_quality: AudioQuality::default(),
            enable_caching: true,
            cache_ttl: Duration::from_secs(3600), // 1 小时
            enable_monitoring: true,
            transcription: TranscriptionConfig::default(),
            synthesis: SynthesisConfig::default(),
            effects: EffectsConfig::default(),
        }
    }
}

/// 文本转语音(tts)配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisConfig {
    /// TTS提供商（openai、azure、google、amazon、kokoro 等）
    pub provider: String,
    /// API的URL
    pub endpoint: Option<String>,
    /// 用于身份验证的API密钥
    pub api_key: Option<String>,
    /// 要使用的语音模型（speaker）
    pub voice: String,
    /// 语音速度（0.25至4.0）
    pub speed: f32,
    /// 输出音频格式
    pub output_format: AudioFormat,
    /// 请求超时（秒）
    pub timeout: u64,
}
impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            endpoint: None,
            api_key: None,
            voice: "alloy".to_string(),
            speed: 1.0,
            output_format: AudioFormat::Mp3,
            timeout: 60,
        }
    }
}

/// 用于读写各种格式的音频编解码器
pub struct AudioCodec;
#[allow(dead_code)]
impl AudioCodec {
    /// 将音频文件解码为AudioData
    pub fn decode_file<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        let path = path.as_ref();
        debug!("Decoding audio file: {}", path.display());

        // 从扩展名检测格式
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| AgentError::invalid_input("Invalid file extension"))?;

        let format = AudioFormat::from_extension(extension).ok_or_else(|| {
            AgentError::invalid_input(format!("Unsupported format: {}", extension))
        })?;

        match format {
            AudioFormat::Wav => Self::decode_wav(path),
            AudioFormat::Mp3 => Self::decode_with_symphonia(path, format),
            AudioFormat::Flac => Self::decode_with_symphonia(path, format),
            AudioFormat::Ogg => Self::decode_with_symphonia(path, format),
            AudioFormat::Aac | AudioFormat::M4a => Self::decode_with_symphonia(path, format),
        }
    }

    /// 从字节缓冲区解码音频
    pub fn decode_bytes(data: &[u8], format: AudioFormat) -> Result<AudioData> {
        debug!(
            "Decoding audio from {} bytes, format: {:?}",
            data.len(),
            format
        );

        match format {
            AudioFormat::Wav => Self::decode_wav_bytes(data),
            AudioFormat::Mp3 => Self::decode_mp3_bytes(data),
            _ => Self::decode_bytes_with_symphonia(data, format),
        }
    }

    /// 将音频数据编码为文件
    pub fn encode_file<P: AsRef<Path>>(
        audio: &AudioData,
        path: P,
        quality: &AudioQuality,
    ) -> Result<()> {
        let path = path.as_ref();
        debug!("Encoding audio to file: {}", path.display());

        // 从扩展名检测格式
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| AgentError::invalid_input("Invalid file extension"))?;

        let format = AudioFormat::from_extension(extension).ok_or_else(|| {
            AgentError::invalid_input(format!("Unsupported format: {}", extension))
        })?;

        match format {
            AudioFormat::Wav => Self::encode_wav(audio, path, quality),
            _ => Err(AgentError::invalid_input(format!(
                "Encoding not supported for format: {:?}",
                format
            ))),
        }
    }

    /// 将音频数据编码为字节
    pub fn encode_bytes(
        audio: &AudioData,
        format: AudioFormat,
        quality: &AudioQuality,
    ) -> Result<Vec<u8>> {
        debug!("Encoding audio to bytes, format: {:?}", format);

        match format {
            AudioFormat::Wav => Self::encode_wav_bytes(audio, quality),
            _ => Err(AgentError::invalid_input(format!(
                "Encoding not supported for format: {:?}",
                format
            ))),
        }
    }

    /// 使用hound解码WAV文件
    fn decode_wav<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| AgentError::invalid_input(format!("Failed to open WAV file: {}", e)))?;

        let spec = reader.spec();
        let samples: Result<Vec<f32>> = match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.map(|sample| sample as f32 / i16::MAX as f32))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| AgentError::invalid_input(format!("Failed to read samples: {}", e))),
            24 => reader
                .samples::<i32>()
                .map(|s| s.map(|sample| sample as f32 / (1 << 23) as f32))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| AgentError::invalid_input(format!("Failed to read samples: {}", e))),
            32 => reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| AgentError::invalid_input(format!("Failed to read samples: {}", e))),
            _ => {
                return Err(AgentError::invalid_input(format!(
                    "Unsupported bit depth: {}",
                    spec.bits_per_sample
                )))
            }
        };

        let samples = samples?;
        Ok(AudioData::new(
            samples,
            spec.sample_rate,
            spec.channels,
            AudioFormat::Wav,
        ))
    }

    /// 从字节中解码WAV
    fn decode_wav_bytes(data: &[u8]) -> Result<AudioData> {
        let cursor = Cursor::new(data);
        let mut reader = hound::WavReader::new(cursor)
            .map_err(|e| AgentError::invalid_input(format!("Failed to read WAV data: {}", e)))?;

        let spec = reader.spec();
        let samples: Vec<f32> = match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.unwrap_or(0) as f32 / i16::MAX as f32)
                .collect(),
            24 => reader
                .samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / (1 << 23) as f32)
                .collect(),
            32 => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
            _ => {
                return Err(AgentError::invalid_input(format!(
                    "Unsupported bit depth: {}",
                    spec.bits_per_sample
                )))
            }
        };

        Ok(AudioData::new(
            samples,
            spec.sample_rate,
            spec.channels,
            AudioFormat::Wav,
        ))
    }

    /// 使用minimp3从字节中解码MP3
    fn decode_mp3_bytes(data: &[u8]) -> Result<AudioData> {
        let mut decoder = minimp3::Decoder::new(data);
        let mut samples = Vec::new();
        let mut sample_rate = 0;
        let mut channels = 0;

        loop {
            match decoder.next_frame() {
                Ok(frame) => {
                    if sample_rate == 0 {
                        sample_rate = frame.sample_rate as u32;
                        channels = frame.channels as u16;
                    }
                    // Convert i16 samples to f32
                    for sample in frame.data {
                        samples.push(sample as f32 / i16::MAX as f32);
                    }
                }
                Err(minimp3::Error::Eof) => break,
                Err(e) => {
                    return Err(AgentError::invalid_input(format!(
                        "MP3 decode error: {:?}",
                        e
                    )))
                }
            }
        }

        if samples.is_empty() {
            return Err(AgentError::invalid_input("No audio data found in MP3"));
        }

        Ok(AudioData::new(
            samples,
            sample_rate,
            channels,
            AudioFormat::Mp3,
        ))
    }

    /// 使用Symphonia解码各种格式
    fn decode_with_symphonia<P: AsRef<Path>>(
        path: P,
        audio_format: AudioFormat,
    ) -> Result<AudioData> {
        let file = File::open(path)
            .map_err(|e| AgentError::invalid_input(format!("Failed to open file: {}", e)))?;

        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        hint.with_extension(audio_format.extension());

        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| AgentError::invalid_input(format!("Failed to probe format: {}", e)))?;

        let mut format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| AgentError::invalid_input("No supported audio tracks found"))?;

        let dec_opts: DecoderOptions = Default::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .map_err(|e| AgentError::invalid_input(format!("Failed to create decoder: {}", e)))?;

        let track_id = track.id;
        let mut samples = Vec::new();
        let mut sample_rate = 0;
        let mut channels = 0;

        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::ResetRequired) => {
                    // Reset decoder and continue
                    decoder.reset();
                    continue;
                }
                Err(SymphoniaError::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => return Err(AgentError::invalid_input(format!("Decode error: {}", e))),
            };

            if packet.track_id() != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    if sample_rate == 0 {
                        let spec = *decoded.spec();
                        sample_rate = spec.rate;
                        channels = spec.channels.count() as u16;
                    }

                    // Convert to f32 samples
                    let mut sample_buf =
                        SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                    sample_buf.copy_interleaved_ref(decoded);
                    samples.extend_from_slice(sample_buf.samples());
                }
                Err(SymphoniaError::IoError(_)) => break,
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(e) => return Err(AgentError::invalid_input(format!("Decode error: {}", e))),
            }
        }

        if samples.is_empty() {
            return Err(AgentError::invalid_input("No audio data decoded"));
        }

        Ok(AudioData::new(samples, sample_rate, channels, audio_format))
    }

    /// 使用Symphonia解码字节
    fn decode_bytes_with_symphonia(data: &[u8], audio_format: AudioFormat) -> Result<AudioData> {
        let data_vec = data.to_vec();
        let cursor = Cursor::new(data_vec);
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
        let mut hint = Hint::new();
        hint.with_extension(audio_format.extension());

        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| AgentError::invalid_input(format!("Failed to probe format: {}", e)))?;

        let mut format_reader = probed.format;
        let track = format_reader
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| AgentError::invalid_input("No supported audio tracks found"))?;

        let dec_opts: DecoderOptions = Default::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .map_err(|e| AgentError::invalid_input(format!("Failed to create decoder: {}", e)))?;

        let track_id = track.id;
        let mut samples = Vec::new();
        let mut sample_rate = 0;
        let mut channels = 0;

        loop {
            let packet = match format_reader.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::ResetRequired) => {
                    decoder.reset();
                    continue;
                }
                Err(SymphoniaError::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(_) => break,
            };

            if packet.track_id() != track_id {
                continue;
            }

            if let Ok(decoded) = decoder.decode(&packet) {
                if sample_rate == 0 {
                    let spec = *decoded.spec();
                    sample_rate = spec.rate;
                    channels = spec.channels.count() as u16;
                }

                let mut sample_buf =
                    SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                sample_buf.copy_interleaved_ref(decoded);
                samples.extend_from_slice(sample_buf.samples());
            }
        }

        if samples.is_empty() {
            return Err(AgentError::invalid_input("No audio data decoded"));
        }

        Ok(AudioData::new(samples, sample_rate, channels, audio_format))
    }

    /// 使用hound对WAV文件进行编码
    fn encode_wav<P: AsRef<Path>>(
        audio: &AudioData,
        path: P,
        quality: &AudioQuality,
    ) -> Result<()> {
        let spec = hound::WavSpec {
            channels: quality.channels,
            sample_rate: quality.sample_rate,
            bits_per_sample: quality.bit_depth,
            sample_format: if quality.bit_depth == 32 {
                hound::SampleFormat::Float
            } else {
                hound::SampleFormat::Int
            },
        };

        let mut writer = hound::WavWriter::create(path, spec).map_err(|e| {
            AgentError::invalid_input(format!("Failed to create WAV writer: {}", e))
        })?;

        // Resample if necessary
        let resampled_audio = if audio.sample_rate != quality.sample_rate {
            audio.resample(quality.sample_rate)?
        } else {
            audio.clone()
        };

        // 根据位深度写入样本
        match quality.bit_depth {
            16 => {
                for sample in &resampled_audio.samples {
                    let sample_i16 = (*sample * i16::MAX as f32) as i16;
                    writer.write_sample(sample_i16).map_err(|e| {
                        AgentError::invalid_input(format!("Failed to write sample: {}", e))
                    })?;
                }
            }
            24 => {
                for sample in &resampled_audio.samples {
                    let sample_i32 = (*sample * (1 << 23) as f32) as i32;
                    writer.write_sample(sample_i32).map_err(|e| {
                        AgentError::invalid_input(format!("Failed to write sample: {}", e))
                    })?;
                }
            }
            32 => {
                for sample in &resampled_audio.samples {
                    writer.write_sample(*sample).map_err(|e| {
                        AgentError::invalid_input(format!("Failed to write sample: {}", e))
                    })?;
                }
            }
            _ => {
                return Err(AgentError::invalid_input(format!(
                    "Unsupported bit depth: {}",
                    quality.bit_depth
                )))
            }
        }

        writer.finalize().map_err(|e| {
            AgentError::invalid_input(format!("Failed to finalize WAV file: {}", e))
        })?;

        Ok(())
    }

    /// 将WAV编码为字节
    fn encode_wav_bytes(audio: &AudioData, quality: &AudioQuality) -> Result<Vec<u8>> {
        let spec = hound::WavSpec {
            channels: quality.channels,
            sample_rate: quality.sample_rate,
            bits_per_sample: quality.bit_depth,
            sample_format: if quality.bit_depth == 32 {
                hound::SampleFormat::Float
            } else {
                hound::SampleFormat::Int
            },
        };

        let mut buffer = Vec::new();
        {
            let cursor = Cursor::new(&mut buffer);
            let mut writer = hound::WavWriter::new(cursor, spec).map_err(|e| {
                AgentError::invalid_input(format!("Failed to create WAV writer: {}", e))
            })?;

            // Resample if necessary
            let resampled_audio = if audio.sample_rate != quality.sample_rate {
                audio.resample(quality.sample_rate)?
            } else {
                audio.clone()
            };

            // 根据位深度写入样本
            match quality.bit_depth {
                16 => {
                    for sample in &resampled_audio.samples {
                        let sample_i16 = (*sample * i16::MAX as f32) as i16;
                        writer.write_sample(sample_i16).map_err(|e| {
                            AgentError::invalid_input(format!("Failed to write sample: {}", e))
                        })?;
                    }
                }
                24 => {
                    for sample in &resampled_audio.samples {
                        let sample_i32 = (*sample * (1 << 23) as f32) as i32;
                        writer.write_sample(sample_i32).map_err(|e| {
                            AgentError::invalid_input(format!("Failed to write sample: {}", e))
                        })?;
                    }
                }
                32 => {
                    for sample in &resampled_audio.samples {
                        writer.write_sample(*sample).map_err(|e| {
                            AgentError::invalid_input(format!("Failed to write sample: {}", e))
                        })?;
                    }
                }
                _ => {
                    return Err(AgentError::invalid_input(format!(
                        "Unsupported bit depth: {}",
                        quality.bit_depth
                    )))
                }
            }

            writer
                .finalize()
                .map_err(|e| AgentError::invalid_input(format!("Failed to finalize WAV: {}", e)))?;
        }

        Ok(buffer)
    }
}
