# Intelligence Agents

## Architecture

The agent layer is intentionally built as an extension of the existing network and runtime layers. It does not replace the DHT, capabilities, or job execution model. Instead, it augments them with a peer-discovered manifest system, a lightweight registry, a DAG-based task decomposition layer, and a scheduling path that can select peers by capability and trust.

## Agent Manifest

Every agent advertises a manifest with:

- stable `id`
- local owner identity and signature
- declared capabilities and tools
- model and resource requirements
- public visibility and remote execution policy
- expiry and trust threshold

The manifest is validated locally before registration. Signatures are checked with the node owner key and expiration is enforced.

## Agent Discovery

Agents publish their identity by capability and role through a registry that indexes local records by capability. This allows the scheduler to match a requirement such as `agent.code.rust` or `agent.verifier` to a set of peers without introducing a central directory.

## Task Graph

Complex work is expressed as an acyclic `AgentTaskGraph` with nodes containing:

- `task_id`
- parent and dependency set
- `required_capabilities`
- priority and deadline
- mutable state
- optional assignment to node and agent
- retry limits

The graph validates dependency completeness, rejects circular references, and preserves a bounded number of tasks.

## Delegation and Scheduling

The scheduler chooses the best peer match based on capability coverage, optional role hints, and a simple score. Delegation depth and fan-out remain bounded so the system avoids unbounded recursion while still decomposing larger objectives.

## System One and model-backend integration

The decision layer is intentionally modeled around a System One-style control loop: evaluate the objective, facts, constraints, and risk, choose the safest next action, and only then delegate or execute work. The implementation is not hardwired to one vendor or model family.

A `SystemOneDecisionEngine` runs with a default safe rule-based fallback and can delegate to any backend implementing the `AgentModelBackend` trait. This keeps an open path for local adapters, `llama.cpp`, Ollama, OpenAI-compatible endpoints, or custom process-based models without replacing the existing network scheduling model.

## Runtime reuse

The agent runtime reuses the existing runtime and job admission path. A task becomes a local `JobRequest`, is admitted through the current queue, and executes with the existing runtime executor semantics. This keeps execution limits, cancellation, deadlines and output bounds consistent with the rest of the network.

## Safety and Trust

Remote messages and remote-advertised agents are treated as untrusted until their signatures and constraints are validated. Resource limits, signature checks, expiry windows, and bounded graph size protect the node from malformed or malicious input.

## MVP status

This first pass includes the core contract and local coordination primitives needed for the network to become agent-aware without breaking the existing protocol or job model:

- manifest validation and signature checks
- agent registry and capability lookup
- DAG task graph validation
- scheduler selection
- state persistence primitives
- runtime reuse for execution

Follow-up stages can extend this foundation with planner swarm orchestration, leases, multi-agent verification, and memory replication while keeping the same trust and execution model.
