//! Builder‐style helper for constructing **Markdown prompts**.
//!
//! Writing verbose Markdown strings inline is tedious and error‐prone.
//! `PromptBuilder` offers a fluent API that lets you focus on the *content*
//! instead of the syntax.  Every method returns `self`, enabling
//! call-chaining:
//!
//! ```rust
//! use artificial_prompt::builder::PromptBuilder;
//!
//! let md = PromptBuilder::new()
//!     .add_section_h1("Mission Briefing")
//!     .add_blank_line()
//!     .add_line("Your objective:")
//!     .add_key_value("Priority", "High")
//!     .add_blank_line()
//!     .add_text_markdown("Remember: **no disintegrations**.")
//!     .finalize();
//!
//! assert!(md.starts_with("# Mission Briefing"));
//! ```
//!
//! The builder performs **no validation** besides `expect`ing that writing to
//! the internal `String` never fails (which it shouldn’t).  It also refrains
//! from smart-formatting to stay predictable—newlines and whitespace are
//! emitted exactly as requested.

use std::fmt::{Display, Write as _};

/// Fluent helper to produce markdown fragments.
///
/// Internally it owns a `String` buffer that grows with each chained call.
/// Once you’re done, call [`Self::finalize`] to obtain the assembled markdown.
pub struct PromptBuilder {
    buffer: String,
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptBuilder {
    /// Create a fresh, empty builder.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Add a level-1 (`#`) heading.
    pub fn add_section_h1(mut self, line: impl Display) -> Self {
        writeln!(self.buffer, "# {line}").expect("failed to write buffer");
        self
    }

    /// Add a level-2 (`##`) heading.
    pub fn add_section_h2(mut self, line: impl Display) -> Self {
        writeln!(self.buffer, "## {line}").expect("failed to write buffer");
        self
    }

    /// Add a plain line of text and a trailing newline.
    pub fn add_line(mut self, line: impl Display) -> Self {
        writeln!(self.buffer, "{line}").expect("failed to write buffer");
        self
    }

    /// Add a bold line (`**text**`) and a trailing newline.
    pub fn add_line_bold(mut self, line: impl Display) -> Self {
        writeln!(self.buffer, "**{line}**").expect("failed to write buffer");
        self
    }

    /// Add a key–value pair in **bold**:
    /// `**Key**: Value`
    pub fn add_key_value(mut self, key: impl Display, value: impl Display) -> Self {
        writeln!(self.buffer, "**{key}**: {value}").expect("failed to write buffer");
        self
    }

    /// Embed a markdown block fenced as `markdown`.
    ///
    /// ```rust
    /// use artificial_prompt::builder::PromptBuilder;
    /// let block = PromptBuilder::new()
    ///     .add_text_markdown("**bold** inside fenced block")
    ///     .finalize();
    /// ```
    pub fn add_text_markdown(self, content: impl Display) -> Self {
        self.add_line("```markdown")
            .add_line(content)
            .add_line("```")
    }

    /// Embed a code block fenced as `json`.
    pub fn add_text_json(self, content: impl Display) -> Self {
        self.add_line("```json").add_line(content).add_line("```")
    }

    /// Embed a code block fenced as `yaml`.
    #[allow(dead_code)]
    pub fn add_text_yaml(self, content: impl Display) -> Self {
        self.add_line("```yaml").add_line(content).add_line("```")
    }

    /// Insert a single blank line.
    pub fn add_blank_line(mut self) -> Self {
        self.buffer.push('\n');
        self
    }

    /// Insert a "---" delimiter.
    pub fn add_delimiter(self) -> Self {
        self.add_line("---")
    }

    /// Insert a horizontal tab character (`\t`).
    ///
    /// Rarely needed, but handy when you want to indent an entire block.
    #[allow(dead_code)]
    pub fn indent(mut self) -> Self {
        self.buffer.push('\t');
        self
    }

    /// Retrieve the accumulated markdown and consume the builder.
    pub fn finalize(self) -> String {
        self.buffer
    }
}
