package com.example.facerecognition.data

import android.content.Context
import android.util.Log
import com.google.gson.*
import java.lang.reflect.Type

data class EmbeddingEntry(
    val id: String,
    val vector: FloatArray
)

data class EmbeddingsStore(
    val version: Int,
    val dimension: Int,
    val count: Int,
    val embeddings: List<EmbeddingEntry>
) {
    private val index: Map<String, FloatArray> = embeddings.associate { it.id to it.vector }
    fun get(id: String): FloatArray? = index[id]
    fun allIds(): Set<String> = index.keys
}

private class EmbeddingEntryDeserializer : JsonDeserializer<EmbeddingEntry> {
    override fun deserialize(json: JsonElement, typeOfT: Type, context: JsonDeserializationContext): EmbeddingEntry {
        val obj = json.asJsonObject
        val id = obj.get("id").asString
        val vecJson = obj.getAsJsonArray("vector")
        val arr = FloatArray(vecJson.size()) { i -> vecJson[i].asFloat }
        return EmbeddingEntry(id, arr)
    }
}

private class EmbeddingsStoreDeserializer : JsonDeserializer<EmbeddingsStore> {
    override fun deserialize(json: JsonElement, typeOfT: Type, context: JsonDeserializationContext): EmbeddingsStore {
        val obj = json.asJsonObject
        val version = obj.get("version").asInt
        val dimension = obj.get("dimension").asInt
        val count = obj.get("count").asInt
        val embArray = obj.getAsJsonArray("embeddings")
        val list = embArray.map { context.deserialize<EmbeddingEntry>(it, EmbeddingEntry::class.java) }
        return EmbeddingsStore(version, dimension, count, list)
    }
}

object EmbeddingsLoader {
    private const val FILE_NAME = "embeddings.json"
    private const val TAG = "EmbeddingsLoader"

    private val gson: Gson = GsonBuilder()
        .registerTypeAdapter(EmbeddingEntry::class.java, EmbeddingEntryDeserializer())
        .registerTypeAdapter(EmbeddingsStore::class.java, EmbeddingsStoreDeserializer())
        .create()

    @Volatile
    private var cached: EmbeddingsStore? = null

    fun load(context: Context): EmbeddingsStore? {
        cached?.let { return it }
        return try {
            val json = context.assets.open(FILE_NAME).bufferedReader().use { it.readText() }
            val store = gson.fromJson(json, EmbeddingsStore::class.java)
            if (store.dimension > 0 && store.embeddings.all { it.vector.size == store.dimension }) {
                cached = store
                Log.d(TAG, "Embeddings chargés: ${store.count} items dimension=${store.dimension}")
                store
            } else {
                Log.e(TAG, "Dimension incohérente dans embeddings.json")
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Erreur chargement embeddings.json: ${e.message}", e)
            null
        }
    }

    fun clearCache() { cached = null }
}
