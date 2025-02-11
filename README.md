# Efficient N-Gram Models for Text Generation in Go: Computational Optimization for Resource-Constrained Systems

![Go Version](https://img.shields.io/badge/Go-1.21+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance implementation of n-gram language models with optimized memory/compute tradeoffs for text generation on low-resource hardware. Implements probabilistic sampling, smoothing techniques, and parallelized corpus processing in Go.

## Technical Overview

N-gram models approximate the probability distribution of text sequences using Markov chain assumptions of order \(n-1\):

\[
P(w_t | w_{t-1}, ..., w_{t-n+1}) = \frac{C(w_{t-n+1}, ..., w_t)}{C(w_{t-n+1}, ..., w_{t-1})}
\]

This implementation focuses on:
- **Memory-efficient backing stores**: Probing hash tables with linear collision resolution for O(1) n-gram lookups
- **Statistical smoothing**: Kneser-Ney interpolation (absolute discounting with lower-order continuation probabilities)
- **Streaming corpus processing**: Goroutine-based parallel tokenization with lock-free aggregation
- **Quantized probability storage**: 16-bit fixed-point representation (10^-4 precision)
- **Efficient sampling**: Vose's alias method O(1) weighted random selection

## Technical Highlights

### Memory-Optimized N-Gram Storage
- **Probabilistic Hash Table**: Open addressing with linear probing (load factor α=0.75)
- **Suffix Array Compression**: Store only suffix words and backoff pointers
- **Trie Pruning**: Remove branches with count < ε (ε=2) during model serialization

```go
type NGramKey [n]uint32 // Hashed n-gram prefix
type NGramEntry struct {
    Suffix    uint32
    Count     uint16
    Backoff   float32
}
