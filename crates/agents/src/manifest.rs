use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use intelligence_protocol::{MetadataEntry, NodeId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MAX_AGENT_CAPABILITIES: usize = 32;
pub const MAX_AGENT_TOOLS: usize = 32;
pub const MAX_AGENT_METADATA: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AgentManifest {
    pub id: String,
    pub name: String,
    pub version: u16,
    pub description: String,
    pub owner: NodeId,
    pub owner_public_key: [u8; 32],
    pub capabilities: Vec<String>,
    pub required_capabilities: Vec<String>,
    pub tools: Vec<String>,
    pub model_requirement: Option<String>,
    pub memory_requirement: u64,
    pub max_concurrent_tasks: u16,
    pub max_task_duration_ms: u64,
    pub max_input_bytes: u32,
    pub max_output_bytes: u32,
    pub public: bool,
    pub accept_remote_tasks: bool,
    pub trust_requirements: u8,
    pub metadata: Vec<MetadataEntry>,
    pub expires_at: u64,
    pub signature: Vec<u8>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ManifestError {
    #[error("agent manifest field {0} is invalid")]
    Invalid(&'static str),
    #[error("agent manifest field {0} exceeds its bound")]
    TooMany(&'static str),
    #[error("agent manifest signature is invalid")]
    Signature,
    #[error("agent manifest encoding failed: {0}")]
    Encoding(String),
}

impl AgentManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.id.is_empty() || self.id.len() > 128 {
            return Err(ManifestError::Invalid("id"));
        }
        if self.name.is_empty() || self.name.len() > 128 {
            return Err(ManifestError::Invalid("name"));
        }
        if self.version == 0 || self.description.len() > 1024 {
            return Err(ManifestError::Invalid("version_or_description"));
        }
        if self.owner == NodeId::default()
            || self.owner != NodeId::from_public_key(&self.owner_public_key)
            || self.expires_at == 0
        {
            return Err(ManifestError::Invalid("owner_or_expiry"));
        }
        if self.capabilities.is_empty()
            || self.capabilities.len() > MAX_AGENT_CAPABILITIES
            || self.required_capabilities.len() > MAX_AGENT_CAPABILITIES
        {
            return Err(ManifestError::TooMany("capabilities"));
        }
        if self.tools.len() > MAX_AGENT_TOOLS || self.metadata.len() > MAX_AGENT_METADATA {
            return Err(ManifestError::TooMany("tools_or_metadata"));
        }
        for capability in self
            .capabilities
            .iter()
            .chain(self.required_capabilities.iter())
            .chain(self.tools.iter())
        {
            if capability.is_empty() || capability.len() > 128 {
                return Err(ManifestError::Invalid("capability_or_tool"));
            }
        }
        if self
            .model_requirement
            .as_ref()
            .is_some_and(|value| value.is_empty() || value.len() > 128)
        {
            return Err(ManifestError::Invalid("model_requirement"));
        }
        if self.max_concurrent_tasks == 0
            || self.max_task_duration_ms == 0
            || self.max_input_bytes == 0
            || self.max_input_bytes as usize > intelligence_protocol::MAX_JOB_INPUT
            || self.max_output_bytes == 0
            || self.max_output_bytes as usize > intelligence_protocol::MAX_JOB_OUTPUT
        {
            return Err(ManifestError::Invalid("resource_limits"));
        }
        for item in &self.metadata {
            if item.key.is_empty() || item.key.len() > 64 || item.value.len() > 256 {
                return Err(ManifestError::Invalid("metadata"));
            }
        }
        if !self.signature.is_empty() && self.signature.len() != 64 {
            return Err(ManifestError::Signature);
        }
        Ok(())
    }

    pub fn signing_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        let mut unsigned = self.clone();
        unsigned.signature.clear();
        postcard::to_allocvec(&unsigned).map_err(|error| ManifestError::Encoding(error.to_string()))
    }

    pub fn verify_signature(&self) -> Result<(), ManifestError> {
        self.validate()?;
        let signature =
            Signature::from_slice(&self.signature).map_err(|_| ManifestError::Signature)?;
        let key = VerifyingKey::from_bytes(&self.owner_public_key)
            .map_err(|_| ManifestError::Signature)?;
        key.verify(&self.signing_bytes()?, &signature)
            .map_err(|_| ManifestError::Signature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn manifest(key: &SigningKey) -> AgentManifest {
        let public_key = key.verifying_key().to_bytes();
        AgentManifest {
            id: "agent.research.general".to_string(),
            name: "General Research Agent".to_string(),
            version: 1,
            description: "Bounded test agent".to_string(),
            owner: NodeId::from_public_key(&public_key),
            owner_public_key: public_key,
            capabilities: vec!["research.text".to_string()],
            required_capabilities: Vec::new(),
            tools: vec!["tool.search".to_string()],
            model_requirement: None,
            memory_requirement: 1024,
            max_concurrent_tasks: 1,
            max_task_duration_ms: 10_000,
            max_input_bytes: 1024,
            max_output_bytes: 1024,
            public: true,
            accept_remote_tasks: true,
            trust_requirements: 0,
            metadata: Vec::new(),
            expires_at: 100,
            signature: Vec::new(),
        }
    }

    #[test]
    fn signed_manifest_verifies_and_tampering_is_rejected() {
        let key = SigningKey::from_bytes(&[7; 32]);
        let mut value = manifest(&key);
        value.signature = key
            .sign(&value.signing_bytes().unwrap())
            .to_bytes()
            .to_vec();
        assert!(value.verify_signature().is_ok());

        value.name.push('!');
        assert_eq!(value.verify_signature(), Err(ManifestError::Signature));
    }
}
