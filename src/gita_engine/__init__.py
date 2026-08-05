"""Gita Engine: a retrieval-grounded reasoning system for the Bhagavad Gita.

This package is organized by pipeline stage (ingestion -> embedding ->
retrieval -> reasoning -> generation -> evaluation), plus cross-cutting
`core` (config/logging) and `db` (persistence) modules. See docs/phases/
for the design rationale behind each stage, written as each phase lands.
"""
