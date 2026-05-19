import streamlit as st
import uuid
from graph import crear_helpdesk, HelpdeskState
from setup_rag import DocumentProcessor
from datetime import datetime
import os

st.set_page_config(
    page_title="Helpdesk 2.0 con RAG y Interrupt Before",
    page_icon="🎧",
    layout="wide"
)

if "helpdesk" not in st.session_state:
    st.session_state.helpdesk = crear_helpdesk()
    st.session_state.tickets = {}

def verificar_rag_setup():
    processor = DocumentProcessor()
    return processor.chroma_path.exists()

def configurar_rag():
    with st.spinner("🔧 Configurando sistema RAG..."):
        processor = DocumentProcessor()
        vectorstore = processor.setup_rag_system(force_rebuild=True)
        return vectorstore is not None

def crear_ticket_id():
    return f"TK-{uuid.uuid4().hex[:6].upper()}"

def procesar_consulta(consulta: str, ticket_id: str):
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
    interrupt_occurred = False
    
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
        
        siguiente = estado_final.next
        if siguiente and len(siguiente) > 0 and "procesar_humano" in str(siguiente):
            interrupt_occurred = True
        
        return estado_final.values, historial_procesamiento, config, interrupt_occurred
        
    except Exception as e:
        st.error(f"Error procesando consulta: {str(e)}")
        return None, [], None, False

def main():
    st.title("🎧 Helpdesk 2.0 Con interrupt_before")
    st.markdown("*Sistema con pausa en nodo esperar_humano*")
    
    rag_configurado = verificar_rag_setup()
    
    with st.sidebar:
        st.header("📊 Panel de Control")
        st.metric("Tickets Activos", len(st.session_state.tickets))
        
        st.subheader("🔍 Estado RAG")
        if rag_configurado:
            st.success("✅ ChromaDB configurado")
        else:
            st.warning("⚠️ RAG no configurado")
            if st.button("🚀 Configurar RAG"):
                if configurar_rag():
                    st.success("✅ RAG configurado")
                    st.rerun()
                else:
                    st.error("❌ Error")
        
        st.subheader("🔄 Flujo v2")
        st.text("""
1. 📝 Consulta → RAG
2. 📊 Clasificación
3. ├─ auto → respuesta_final → END
4. └─ escalated → ⏸️ ESPERA
5.    (interrupt_before activa)
6.    👨‍💼 Revisa/Edita respuesta RAG
7.    → procesar_humano → END
        """)
        
        st.subheader("⚙️ Config")
        if st.button("🔄 Reconfigurar RAG"):
            configurar_rag()
            st.rerun()
        
        if st.button("🗑️ Limpiar Tickets"):
            st.session_state.tickets = {}
            st.rerun()
    
    if not rag_configurado:
        st.warning("⚠️ Configura el RAG primero")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Nueva Consulta")
        
        with st.expander("💡 Ejemplos"):
            ejemplos = [
                "No puedo resetear mi contraseña",
                "Error 500 en la aplicación",
                "¿Cómo cancelo mi suscripción?",
                "La aplicación va muy lenta"
            ]
            for ejemplo in ejemplos:
                if st.button(f"📋 {ejemplo}", key=f"ej_{ejemplo}"):
                    st.session_state.consulta_ejemplo = ejemplo
        
        with st.form("nueva_consulta"):
            usuario = st.text_input("👤 Usuario", placeholder="tu@email.com")
            
            consulta_inicial = st.session_state.get("consulta_ejemplo", "")
            consulta = st.text_area(
                "💬 Describe tu problema",
                value=consulta_inicial,
                placeholder="...",
                height=100
            )
            
            submitted = st.form_submit_button("🚀 Enviar")
            
            if submitted and consulta.strip():
                if "consulta_ejemplo" in st.session_state:
                    del st.session_state.consulta_ejemplo
                
                ticket_id = crear_ticket_id()
                
                with st.spinner("🔄 Procesando..."):
                    resultado, historial, config, interrupt_occurred = procesar_consulta(consulta, ticket_id)
                
                if resultado:
                    st.session_state.tickets[ticket_id] = {
                        "usuario": usuario,
                        "consulta": consulta,
                        "resultado": resultado,
                        "historial": historial,
                        "config": config,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "interrupt_occurred": interrupt_occurred
                    }
                    
                    st.success(f"✅ Ticket {ticket_id}" + (" - ⏸️ Esperando humano" if interrupt_occurred else ""))
                    st.rerun()
    
    with col2:
        st.subheader("🎫 Tickets")
        
        if not st.session_state.tickets:
            st.info("No hay tickets")
        else:
            for ticket_id, ticket_data in reversed(list(st.session_state.tickets.items())):
                with st.expander(f"🎫 {ticket_id} - {ticket_data['timestamp']}", expanded=True):
                    st.markdown(f"**👤** {ticket_data['usuario']}")
                    st.markdown(f"**💬** {ticket_data['consulta'][:80]}...")
                    
                    st.subheader("🔄 Procesamiento:")
                    for paso in ticket_data['historial']:
                        st.text(paso)
                    
                    resultado = ticket_data['resultado']
                    
                    if resultado.get('categoria'):
                        st.markdown(f"**📂 Categoría:** {resultado['categoria']}")
                    
                    if resultado.get('confianza', 0) > 0:
                        confianza = resultado['confianza']
                        st.markdown(f"**🎯 Confianza RAG:** {confianza:.2f}")
                        color = "green" if confianza >= 0.65 else "orange" if confianza >= 0.4 else "red"
                        st.progress(confianza)
                        
                        if resultado.get('fuentes'):
                            st.markdown(f"**📚 Fuentes:** {', '.join(resultado['fuentes'])}")
                    
                    # CASO: Escalado con intervención humana
                    if resultado.get('requiere_humano') and not resultado.get('respuesta_final'):
                        st.warning("⏸️ **INTERRUMPIDO** - Esperando revisión humana")
                        
                        # Mostrar respuesta RAG para editar
                        respuesta_rag = resultado.get('respuesta_rag', '')
                        if respuesta_rag:
                            st.info(f"**📋 Respuesta RAG generada:**\n{respuesta_rag}")
                        
                        # Campo para editar la respuesta
                        respuesta_editada = st.text_area(
                            "✍️ Edita/Aprueba la respuesta:",
                            value=respuesta_rag,
                            key=f"resp_{ticket_id}",
                            height=150
                        )
                        
                        col_btn1, col_btn2 = st.columns(2)
                        
                        with col_btn1:
                            if st.button(f"✅ Aprobar/Enviar", key=f"btn_{ticket_id}"):
                                if respuesta_editada.strip():
                                    config = ticket_data['config']
                                    st.session_state.helpdesk.update_state(
                                        config,
                                        {"respuesta_humano": respuesta_editada}
                                    )
                                    
                                    for chunk in st.session_state.helpdesk.stream(None, config=config, stream_mode="updates"):
                                        for nodo, salida in chunk.items():
                                            if "historial" in salida and salida["historial"]:
                                                ticket_data['historial'].extend(salida["historial"])
                                    
                                    estado_final = st.session_state.helpdesk.get_state(config)
                                    ticket_data['resultado'] = estado_final.values
                                    
                                    st.success("✅ Respuesta enviada")
                                    st.rerun()
                                else:
                                    st.warning("⚠️ Escribe una respuesta")
                        
                        with col_btn2:
                            if st.button(f"❌ Descartar", key=f"desc_{ticket_id}"):
                                st.warning("Ticket descartado - réanudando sin intervención")
                                config = ticket_data['config']
                                st.session_state.helpdesk.update_state(
                                    config,
                                    {"respuesta_humano": "Descartado por agente"}
                                )
                                
                                for chunk in st.session_state.helpdesk.stream(None, config=config, stream_mode="updates"):
                                    for nodo, salida in chunk.items():
                                        if "historial" in salida and salida["historial"]:
                                            ticket_data['historial'].extend(salida["historial"])
                                
                                estado_final = st.session_state.helpdesk.get_state(config)
                                ticket_data['resultado'] = estado_final.values
                                st.rerun()
                    
                    # Ticket resuelto
                    elif resultado.get('respuesta_final'):
                        st.success("✅ **RESUELTO**")
                        st.markdown("**💬 Respuesta:**")
                        st.info(resultado['respuesta_final'])
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("🎯 Confianza", f"{resultado.get('confianza', 0):.2f}")
                        with col_m2:
                            st.metric("🔍 Fuentes", len(resultado.get('fuentes', [])))
                        with col_m3:
                            resolucion = "RAG" if not resultado.get('requiere_humano') else "Humano+"
                            st.metric("🤖 Resuelto por", resolucion)
    
    st.markdown("---")
    if st.session_state.tickets:
        total = len(st.session_state.tickets)
        resueltos = sum(1 for t in st.session_state.tickets.values() if t['resultado'].get('respuesta_final'))
        pendientes = total - resueltos
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total", total)
        with col2:
            st.metric("✅ Resueltos", resueltos)
        with col3:
            st.metric("⏳ Pendientes", pendientes)

if __name__ == "__main__":
    main()