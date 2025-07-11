//! Helpers for turning Rust type information into JSON Schema that can be
//! shipped alongside a prompt. The JSON is produced with [`schemars`] and
//! can be forwarded to providers that support structured / function-calling
//! responses (e.g. OpenAI’s *response_format = json_schema*).
//!
//! The abstraction is intentionally **very small**: if you need a more
//! sophisticated setup (e.g. inline- vs. $ref-based schemas, custom
//! serialization logic) you can always bypass this helper and build the
//! schema manually.

use schemars::{r#gen::SchemaSettings, JsonSchema, SchemaGenerator};
use serde_json::{self, Value};

/// Generate a JSON Schema for the given `T` **inline**, i.e. without
/// `$ref` pointers to external definitions.
///
/// This is sufficient for most LLM providers, which currently expect the
/// entire schema object inside a single request.
///
/// # Panics
///
/// This function panics only if the resulting [`RootSchema`] cannot be
/// serialized into valid JSON – which should never happen as long as
/// [`schemars`] works correctly.
///
/// # Example
///
/// ```
/// use artificial_core::schema_util::derive_response_schema;
/// use schemars::JsonSchema;
///
/// #[derive(JsonSchema)]
/// struct Foo { bar: String }
///
/// let schema = derive_response_schema::<Foo>();
/// println!("{}", serde_json::to_string_pretty(&schema).unwrap());
/// ```
pub fn derive_response_schema<T>() -> Value
where
    T: JsonSchema + 'static,
{
    // We want the schema fully inlined to avoid `$ref`s that some providers
    // may not resolve correctly.
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;

    let generator = SchemaGenerator::new(settings);
    let root = generator.into_root_schema_for::<T>();

    serde_json::to_value(root).expect("generated schema should be serialisable")
}
