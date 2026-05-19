import streamlit as st
import uuid
from graph import crear_helpdesk, HelpdeskState
from setup_rag import DocumentProcessor
from datetime import datetime
import os

st.set_page_config(
    page_title="Helpdesk 2.0 with RAG",
    page_icon="🎧",
    layout="wide"
)

if "helpdesk" not in st.session_state:
    st.session_state.helpdesk = crear_helpdesk()
    st.session_state.tickets = {}

def verificar_rag_setup():
    """Verifica si el sistema RAG está configurado."""
    processor = DocumentProcessor()
    return processor.chroma_path.exists()

def configurar_rag():
    """Configura el sistema RAG."""
    with st.spinner("🔧 Configuring RAG system..."):
        processor = DocumentProcessor()
        vectorstore = processor.setup_rag_system(force_rebuild=True)
        return vectorstore is not None

def crear_ticket_id():
    """Genera un ID único para el ticket."""
    return f"TK-{uuid.uuid4().hex[:6].upper()}"

def procesar_consulta(consulta: str, ticket_id: str):
    """Procesa una consulta nueva."""
    estado_inicial = HelpdeskState(
        consulta=consulta,
        categoria="",
        respuesta_rag=None,
        confianza=0.0,
        fuentes=[],
        requiere_humano=False,
        respuesta_humano=None,
        respuesta_final=None,
        historial=[]
    )
    
    config = {"configurable": {"thread_id": ticket_id}}
    
    historial_procesamiento = []
    
    try:
        for chunk in st.session_state.helpdesk.stream(
            estado_inicial, 
            config=config, 
            stream_mode="updates"
        ):
            for nodo, salida in chunk.items():
                if "historial" in salida and salida["historial"]:
                    historial_procesamiento.extend(salida["historial"])
        
        estado_final = st.session_state.helpdesk.get_state(config)
        
        return estado_final.values, historial_procesamiento, config
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None, [], None

def main():
    """Aplicación principal."""
    st.title("🎧 Helpdesk 2.0 with RAG + ChromaDB")
    st.markdown("*Intelligent system with LangGraph and vector search*")
    
    rag_configurado = verificar_rag_setup()
    
    with st.sidebar:
        st.header("📊 Control Panel")
        st.metric("Active Tickets", len(st.session_state.tickets))
        
        st.subheader("🔍 RAG Status")
        if rag_configurado:
            st.success("✅ ChromaDB configured")
        else:
            st.warning("⚠️ RAG not configured")
            if st.button("🚀 Configure RAG"):
                if configurar_rag():
                    st.success("✅ RAG configured successfully")
                    st.rerun()
                else:
                    st.error("❌ Error configuring RAG")
        
        st.subheader("🔄 System Flow")
        st.text("""
1. 📝 User submits query
2. 🤖 Automatic classification
3. 🔍 RAG vector search
4. 📊 Confidence evaluation
5. 👨‍💼 Escalation if needed
6. ✅ Final response
        """)
        
        st.subheader("⚙️ Configuration")
        if st.button("🔄 Reconfigure RAG"):
            if configurar_rag():
                st.success("✅ RAG reconfigured")
                st.rerun()
        
        if st.button("🗑️ Clear Tickets"):
            st.session_state.tickets = {}
            st.rerun()
    
    if not rag_configurado:
        st.warning("⚠️ RAG system not configured. Use the sidebar button to configure it.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 New Query")
        
        with st.expander("💡 Query Examples"):
            ejemplos = [
                "I can't reset my password",
                "Error 500 in the application",
                "How do I cancel my subscription?",
                "The application is very slow",
                "Billing problems"
            ]
            for ejemplo in ejemplos:
                if st.button(f"📋 {ejemplo}", key=f"ej_{ejemplo}"):
                    st.session_state.consulta_ejemplo = ejemplo
        
        with st.form("nueva_consulta"):
            usuario = st.text_input("👤 User", placeholder="your@email.com")
            
            consulta_inicial = st.session_state.get("consulta_ejemplo", "")
            consulta = st.text_area(
                "💬 Problem Description",
                value=consulta_inicial,
                placeholder="Describe your query or problem here...",
                height=100
            )
            
            submitted = st.form_submit_button("🚀 Submit Query")
            
            if submitted and consulta.strip():
                if "consulta_ejemplo" in st.session_state:
                    del st.session_state.consulta_ejemplo
                
                ticket_id = crear_ticket_id()
                
                with st.spinner("🔄 Processing query..."):
                    resultado, historial, config = procesar_consulta(consulta, ticket_id)
                
                if resultado:
                    st.session_state.tickets[ticket_id] = {
                        "usuario": usuario,
                        "consulta": consulta,
                        "resultado": resultado,
                        "historial": historial,
                        "config": config,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    
                    st.success(f"✅ Ticket {ticket_id} created")
                    st.rerun()
    
    with col2:
        st.subheader("🎫 Recent Tickets")
        
        if not st.session_state.tickets:
            st.info("No active tickets")
        else:
            for ticket_id, ticket_data in reversed(list(st.session_state.tickets.items())):
                with st.expander(f"🎫 {ticket_id} - {ticket_data['timestamp']}", expanded=True):
                    st.markdown(f"**👤 User:** {ticket_data['usuario']}")
                    st.markdown(f"**💬 Query:** {ticket_data['consulta'][:100]}...")
                    
                    resultado = ticket_data['resultado']
                    
                    st.subheader("🔄 Processing:")
                    for paso in ticket_data['historial']:
                        st.text(paso)
                    
                    if resultado.get('categoria'):
                        st.markdown(f"**📂 Category:** {resultado['categoria']}")
                    
                    if resultado.get('confianza', 0) > 0:
                        confidence = resultado['confianza']
                        st.markdown(f"**🎯 RAG Confidence:** {confidence:.2f}")
                        
                        progress_color = "green" if confidence >= 0.65 else "orange" if confidence >= 0.4 else "red"
                        st.progress(confidence)
                        
                        if resultado.get('fuentes'):
                            st.markdown(f"**📚 Sources:** {', '.join(resultado['fuentes'])}")
                    
                    if resultado.get('requiere_humano') and not resultado.get('respuesta_final'):
                        st.warning("👨‍💼 Requires human intervention")
                        
                        if resultado.get('respuesta_rag'):
                            with st.expander("📋 Context for agent"):
                                st.text(resultado['respuesta_rag'])
                        
                        respuesta_humano = st.text_area(
                            "✍️ Agent response:",
                            key=f"respuesta_{ticket_id}",
                            height=100,
                            placeholder="Write the response for the user..."
                        )
                        
                        col_btn1, col_btn2 = st.columns(2)
                        
                        with col_btn1:
                            if st.button(f"💾 Submit Response", key=f"btn_{ticket_id}"):
                                if respuesta_humano.strip():
                                    config = ticket_data['config']
                                    st.session_state.helpdesk.update_state(
                                        config,
                                        {"respuesta_humano": respuesta_humano}
                                    )
                                    
                                    for chunk in st.session_state.helpdesk.stream(None, config=config, stream_mode="updates"):
                                        for nodo, salida in chunk.items():
                                            if "historial" in salida and salida["historial"]:
                                                ticket_data['historial'].extend(salida["historial"])
                                    
                                    estado_final = st.session_state.helpdesk.get_state(config)
                                    ticket_data['resultado'] = estado_final.values
                                    
                                    st.success("✅ Response processed")
                                    st.rerun()
                                else:
                                    st.warning("⚠️ Write a response before submitting")
                        
                        with col_btn2:
                            if st.button(f"🔄 Use RAG", key=f"rag_{ticket_id}"):
                                respuesta_rag = resultado.get('respuesta_rag', '')
                                config = ticket_data['config']
                                st.session_state.helpdesk.update_state(
                                    config,
                                    {"respuesta_humano": respuesta_rag}
                                )
                                
                                for chunk in st.session_state.helpdesk.stream(None, config=config, stream_mode="updates"):
                                    for nodo, salida in chunk.items():
                                        if "historial" in salida and salida["historial"]:
                                            ticket_data['historial'].extend(salida["historial"])
                                
                                estado_final = st.session_state.helpdesk.get_state(config)
                                ticket_data['resultado'] = estado_final.values
                                
                                st.success("✅ RAG response applied")
                                st.rerun()
                    
                    elif resultado.get('respuesta_final'):
                        st.success("✅ Ticket Resolved")
                        st.markdown("**💬 Response:**")
                        
                        respuesta = resultado['respuesta_final']
                        st.info(respuesta)
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("🎯 Confidence", f"{resultado.get('confianza', 0):.2f}")
                        with col_m2:
                            st.metric("🔍 Sources", len(resultado.get('fuentes', [])))
                        with col_m3:
                            resolucion = "RAG" if not resultado.get('requiere_humano') else "Human"
                            st.metric("🤖 Resolved by", resolucion)
    
    st.markdown("---")
    if st.session_state.tickets:
        total_tickets = len(st.session_state.tickets)
        resueltos_rag = sum(1 for t in st.session_state.tickets.values() 
                           if t['resultado'].get('respuesta_final') and not t['resultado'].get('requiere_humano'))
        resueltos_humano = sum(1 for t in st.session_state.tickets.values() 
                              if t['resultado'].get('respuesta_final') and t['resultado'].get('requiere_humano'))
        pendientes = total_tickets - resueltos_rag - resueltos_humano
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("📊 Total Tickets", total_tickets)
        with col_stat2:
            st.metric("🤖 Resolved by RAG", resueltos_rag)
        with col_stat3:
            st.metric("👨‍💼 Resolved by Human", resueltos_humano)
        with col_stat4:
            st.metric("⏳ Pending", pendientes)
    
    st.markdown(
        """
        <div style='text-align: center'>
            <small>🚀 Powered by LangGraph | 🔍 ChromaDB | 🔄 Streaming | 💾 Checkpointing | 👨‍💼 Human-in-the-Loop</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()