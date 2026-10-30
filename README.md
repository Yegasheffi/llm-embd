# llm-embd
一个常驻本地的**增量 embedding 索引层**：文件变了以后，如何只重算真正变化的 chunk，并把结果通过一个稳定的本地 API 交给任何 App。
