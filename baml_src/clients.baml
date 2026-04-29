client<llm> Model {
  provider "openai-generic"
  retry_policy RisikoRetryPolicy
  options {
    base_url env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1"
    model qwen3.5:4b
    // Optimize for speed
    temperature 0.1
    max_tokens 1024
  }
}

// Retry policy for robustness
retry_policy RisikoRetryPolicy {
  max_retries 2
  strategy {
    type exponential_backoff
    delay_ms 100
    multiplier 1.5
    max_delay_ms 2000
  }
}
