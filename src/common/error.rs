use thiserror::Error;

/// Result type alias for the agent system
pub type Result<T> = std::result::Result<T, AgentError>;

/// Comprehensive error types for the agent system
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum AgentError {
    /// Anthropic API related errors
    #[error("Anthropic API error: {message}")]
    AnthropicApi { message: String },

    /// HTTP request errors
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// Memory system errors
    #[error("Memory error: {message}")]
    Memory { message: String },

    /// Tool execution errors
    #[error("Tool error: {tool_name}: {message}")]
    Tool { tool_name: String, message: String },

    /// File system errors
    #[error("File system error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid input errors
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    /// Authentication errors
    #[error("Authentication error: {message}")]
    Authentication { message: String },

    /// Rate limiting errors
    #[error("Rate limit exceeded: {message}")]
    RateLimit { message: String },

    /// Plugin system errors
    #[error("Plugin error: {message}")]
    Plugin { message: String },

    /// Validation errors
    #[error("Validation error: {message}")]
    Validation { message: String },

    /// Security errors
    #[error("Security error: {message}")]
    Security { message: String },

    /// DSPy module errors
    #[error("DSPy module error: {module_name}: {message}")]
    DspyModule {
        module_name: String,
        message: String,
    },

    /// DSPy signature errors
    #[error("DSPy signature error: {message}")]
    DspySignature { message: String },

    /// DSPy optimization errors
    #[error("DSPy optimization error: {strategy}: {message}")]
    DspyOptimization { strategy: String, message: String },

    /// DSPy evaluation errors
    #[error("DSPy evaluation error: {metric}: {message}")]
    DspyEvaluation { metric: String, message: String },

    /// DSPy compilation errors
    #[error("DSPy compilation error: {phase}: {message}")]
    DspyCompilation { phase: String, message: String },

    /// Generic errors
    #[error("Agent error: {0}")]
    Generic(#[from] anyhow::Error),
}

impl AgentError {
    /// Create a new tool error
    pub fn tool<S: Into<String>>(tool_name: S, message: S) -> Self {
        Self::Tool {
            tool_name: tool_name.into(),
            message: message.into(),
        }
    }

    /// Create a new invalid input error
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Create a new authentication error
    pub fn authentication<S: Into<String>>(message: S) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }

    /// Create a new kokoro error
    #[allow(dead_code)]
    pub fn kokoro<S: Into<String>>(message: S) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }
}

#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum TtsError {
    /// TTS初始化相关错误
    #[error("TTS init error: {message}")]
    Init { message: String },
}
impl TtsError {
    /// 创建新的TTS初始化相关错误
    #[allow(dead_code)]
    pub fn init<S: Into<String>>(message: S) -> Self {
        Self::Init {
            message: message.into(),
        }
    }
}
