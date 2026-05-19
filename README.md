# Helpdesk 2.0 con RAG

Sistema inteligente de helpdesk basado en LangGraph con búsqueda vectorial (RAG) usando ChromaDB.

## Descripción

Helpdesk 2.0 es un sistema de soporte técnico inteligente que utiliza:
- **LangGraph** para orquestar el flujo de tickets
- **ChromaDB** como base de datos vectorial para búsqueda semántica
- **MultiQueryRetriever** para mejorar la precisión de las búsquedas
- **Human-in-the-Loop** para escalado a agentes humanos
- **Checkpointing** con SQLite para persistencia de estado

## Arquitectura

```
Usuario → Clasificación → RAG (ChromaDB) → Evaluación de Confianza
                                                  ↓
                              Confianza Alta → Respuesta Automática
                              Confianza Baja → Escalado Humano
```

## Técnicas Utilizadas

### 1. Retrieval Augmented Generation (RAG)
- Búsqueda vectorial en base de conocimiento
- MultiQueryRetriever para mayor cobertura
- Cálculo de confianza basado en relevancia

### 2. LangGraph State Management
- Estado tipado con `TypedDict`
- Checkpointing con SQLite
- Interrupciones para espera de intervención humana
- Streaming de eventos

### 3. Human-in-the-Loop
- Pausa del grafo para espera de respuesta
- Actualización de estado con input humano
- Continuación del flujo post-intervención

## Estructura del Proyecto

```
helpdesk_system/
├── app.py              # Aplicación Streamlit
├── graph.py            # Grafo de LangGraph
├── rag_system.py       # Sistema RAG con ChromaDB
├── setup_rag.py        # Configuración del vectorstore
├── config.py           # Configuración
├── docs/               # Documentación base de conocimiento
│   ├── faq.md
│   ├── manual_usuario.md
│   └── guia_resolucion_problemas.md
└── chroma_db/          # Vectorstore (git ignored)
```

## Requisitos

- Python 3.10+
- OpenAI API Key

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Crear archivo `.env` con tu API key:

```
OPENAI_API_KEY=sk-your-key-here
```

## Uso

1. **Iniciar la aplicación:**
```bash
streamlit run app.py
```

2. **Configurar RAG:**
   - La primera vez, click en "Configurar RAG" en el sidebar
   - Esto cargará los documentos de `docs/` al vectorstore

3. **Crear tickets:**
   - Ingresar consulta del problema
   - El sistema clasificará automáticamente
   - Respuesta automática o escalado a humano

4. **Intervenir en tickets:**
   - Tickets escalados muestran contexto RAG
   - Escribir respuesta o usar respuesta RAG como base

## Flujo del Sistema

1. 📝 Usuario envía consulta
2. 🤖 Clasificación automática (automatico/escalado)
3. 🔍 Búsqueda vectorial RAG
4. 📊 Evaluación de confianza (>0.6 = automático)
5. 👨‍💼 Escalado si confianza baja
6. ✅ Respuesta final