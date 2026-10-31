package com.pixel10.ai.server

import com.google.gson.JsonDeserializationContext
import com.google.gson.JsonDeserializer
import com.google.gson.JsonElement
import com.google.gson.JsonNull
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive
import com.google.gson.JsonSerializationContext
import com.google.gson.JsonSerializer
import com.google.gson.annotations.JsonAdapter
import com.google.gson.annotations.SerializedName
import java.lang.reflect.Type

/**
 * Request/response models for the AI API.
 * Follows the OpenAI Chat Completions schema for easy integration with
 * any OpenAI-compatible client (OpenClaw, LM Studio, Open WebUI, Agora, etc.).
 *
 * Tool/function calling is fully supported so coding agents can invoke
 * tools (read_file, run_shell, etc.) through the standard OpenAI tool-use flow.
 */

// ── Custom Gson Adapters (Agora / OpenAI Multimodal Compatibility) ────────────

/**
 * Handles both classic string content ("content": "hello") and OpenAI/Agora
 * array content ("content": [{"type": "text", "text": "hello"}]).
 */
class FlexibleContentAdapter : JsonDeserializer<String?>, JsonSerializer<String?> {
    override fun deserialize(
        json: JsonElement?,
        typeOfT: Type?,
        context: JsonDeserializationContext?
    ): String? {
        if (json == null || json.isJsonNull) return null
        if (json.isJsonPrimitive) return json.asString
        if (json.isJsonArray) {
            return json.asJsonArray.mapNotNull { element ->
                when {
                    element.isJsonPrimitive -> element.asString
                    element.isJsonObject -> {
                        val obj = element.asJsonObject
                        when {
                            obj.has("text") && !obj.get("text").isJsonNull ->
                                obj.get("text").asString
                            obj.has("content") && !obj.get("content").isJsonNull ->
                                obj.get("content").asString
                            else -> null
                        }
                    }
                    else -> null
                }
            }.joinToString("\n")
        }
        if (json.isJsonObject) {
            val obj = json.asJsonObject
            if (obj.has("text") && !obj.get("text").isJsonNull) {
                return obj.get("text").asString
            }
        }
        return json.toString()
    }

    override fun serialize(
        src: String?,
        typeOfSrc: Type?,
        context: JsonSerializationContext?
    ): JsonElement {
        return if (src == null) JsonNull.INSTANCE else JsonPrimitive(src)
    }
}

/**
 * Prevents crashes if a client sends tool_choice as a JSON object instead of a string.
 */
class FlexibleToolChoiceAdapter : JsonDeserializer<String?>, JsonSerializer<String?> {
    override fun deserialize(
        json: JsonElement?,
        typeOfT: Type?,
        context: JsonDeserializationContext?
    ): String? {
        if (json == null || json.isJsonNull) return null
        if (json.isJsonPrimitive) return json.asString
        return json.toString()
    }

    override fun serialize(
        src: String?,
        typeOfSrc: Type?,
        context: JsonSerializationContext?
    ): JsonElement {
        return if (src == null) JsonNull.INSTANCE else JsonPrimitive(src)
    }
}

// ── Requests ──────────────────────────────────────────────────────────────────

data class ChatRequest(
    val model: String = "pixel10",
    val messages: List<Message> = emptyList(),
    val prompt: String? = null,
    val max_tokens: Int = 8192,
    val temperature: Float = 0.7f,
    val stream: Boolean = false,
    /** Tool/function definitions available to the model. */
    val tools: List<Tool>? = null,
    /** "auto" | "none" | "required" — defaults to "auto" when tools are provided. */
    @JsonAdapter(FlexibleToolChoiceAdapter::class)
    val tool_choice: String? = null
)

data class Message(
    val role: String = "user",
    /** Text content. Null when role=assistant and the model is calling a tool. */
    @JsonAdapter(FlexibleContentAdapter::class)
    val content: String? = null,
    /** Set by the model when it wants to call one or more tools. */
    val tool_calls: List<ToolCall>? = null,
    /** Set on role=tool messages — references the tool_call.id being responded to. */
    val tool_call_id: String? = null
)

// ── Tool / Function Calling ───────────────────────────────────────────────────

data class Tool(
    val type: String = "function",
    val function: ToolFunction
)

data class ToolFunction(
    val name: String,
    val description: String = "",
    /** JSON Schema object describing the function parameters. */
    val parameters: JsonObject? = null
)

data class ToolCall(
    val id: String,
    val type: String = "function",
    val function: FunctionCallDetail
)

data class FunctionCallDetail(
    val name: String,
    /** Arguments as a JSON-encoded string (matches OpenAI spec). */
    val arguments: String
)

// ── Responses ─────────────────────────────────────────────────────────────────

data class ChatResponse(
    val id: String,
    @SerializedName("object")
    val objectType: String = "chat.completion",
    val created: Long = System.currentTimeMillis() / 1000,
    val model: String = "pixel10",
    val choices: List<Choice>,
    val usage: Usage
)

data class Choice(
    val index: Int = 0,
    val message: Message,
    /** "stop" | "tool_calls" | "length" */
    val finish_reason: String = "stop"
)

data class Usage(
    val prompt_tokens: Int,
    val completion_tokens: Int,
    val total_tokens: Int
)

// ── Streaming ─────────────────────────────────────────────────────────────────

data class StreamChunk(
    val id: String,
    @SerializedName("object")
    val objectType: String = "chat.completion.chunk",
    val created: Long = System.currentTimeMillis() / 1000,
    val model: String = "pixel10-fast",
    val choices: List<StreamChoice>
)

data class StreamChoice(
    val index: Int = 0,
    val delta: Delta,
    val finish_reason: String? = null
)

data class Delta(
    val role: String? = null,
    val content: String? = null
)

// ── Models List ───────────────────────────────────────────────────────────────

data class ModelInfo(
    val id: String,
    @SerializedName("object")
    val objectType: String = "model",
    val owned_by: String = "local-device",
    val description: String = "",
    /** Input context window in tokens. */
    val context_length: Int = 1_000_000,
    /** Maximum output tokens. */
    val max_output_tokens: Int = 8192
)

data class ModelList(
    @SerializedName("object")
    val objectType: String = "list",
    val data: List<ModelInfo> = listOf(
        ModelInfo(
            id = "pixel10",
            description = "Gemini Nano on Tensor G5 — fully on-device, private, tool calling supported.",
            context_length = 4096,
            max_output_tokens = 1024
        )
    )
)

// ── Health / Errors ───────────────────────────────────────────────────────────

data class ErrorResponse(
    val error: ErrorDetail
)

data class ErrorDetail(
    val message: String,
    val type: String = "server_error",
    val code: Int = 500
)

data class ServerStatus(
    val status: String,
    val model: String,
    val device: String,
    val uptime_seconds: Long,
    val requests_served: Long,
    val endpoints: List<String> = listOf(
        "POST /v1/chat/completions  (tool calling + streaming)",
        "POST /v1/completions",
        "GET  /v1/models",
        "GET  /v1/agent            (system prompt + tool definitions)",
        "GET  /health",
        "GET  /"
    )
)
