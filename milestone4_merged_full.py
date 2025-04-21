import streamlit as st
import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
import tempfile
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import PyPDF2
import fitz  # PyMuPDF
from urllib.parse import urlparse
import time
import re
import base64

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client (use environment variable for security)
api_key = os.getenv("OPENAI_API_KEY", "API_KEY_HERE")
client = OpenAI(api_key=api_key)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_base_loaded" not in st.session_state:
    st.session_state.knowledge_base_loaded = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "user_context" not in st.session_state:
    st.session_state.user_context = {
        "name": "",
        "role": "",
        "department": "",
        "career_interests": "",
        "skills_to_develop": []
    }
if "conversation_stage" not in st.session_state:
    st.session_state.conversation_stage = "initial"
if "pdf_url" not in st.session_state:
    st.session_state.pdf_url = "https://s3.us-west-1.amazonaws.com/valleywater.org.us-west-1/s3fs-public/Employees%20Association%20MOU%202022-2025.docx%20%283%29.pdf"
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "match_analysis" not in st.session_state:
    st.session_state.match_analysis = None
if "resource_texts" not in st.session_state:
    st.session_state.resource_texts = []

# Extract text from PDF using PyMuPDF (faster and more reliable)
def extract_text(uploaded_file):
    try:
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        file_text = ""
        for i in range(len(pdf)):
            page = pdf.load_page(i)
            file_text += page.get_text("text")
        
        return file_text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

# Extract text from PDF using PyPDF2 (alternative method)
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

# Extract text from PDF URL
def extract_text_from_pdf_url(url):
    try:
        logging.info(f"Attempting to download PDF from {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Create a BytesIO object
        pdf_file = BytesIO(response.content)
        
        # First try PyMuPDF for better results
        try:
            return extract_text(pdf_file)
        except:
            # Fall back to PyPDF2
            pdf_file.seek(0)  # Reset buffer position
            return extract_text_from_pdf(pdf_file)
    except Exception as e:
        logging.error(f"Error extracting text from PDF URL: {e}")
        return None

# Extract text from webpage
def extract_text_from_webpage(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up text (remove extra whitespace)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        logging.error(f"Error extracting text from webpage: {e}")
        return None

# Process documents and create vector store
def create_vector_store(text, source_name):
    try:
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
        )
        
        # Create document object
        doc = Document(page_content=text, metadata={"source": source_name})
        docs = text_splitter.split_documents([doc])
        
        # Create vector store
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vector_store = FAISS.from_documents(docs, embeddings)
        
        logging.info(f"Vector store created successfully from {source_name}")
        return vector_store
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        return None

# Retrieve relevant policy information based on query
def retrieve_policy_info(query, vector_store, top_k=5):
    try:
        if not vector_store:
            return "Knowledge base not available."
        
        results = vector_store.similarity_search(query, k=top_k)
        context = "\n\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        logging.error(f"Error retrieving policy info: {e}")
        return "Unable to retrieve policy information."

# Extract user context from conversation
def extract_user_context(user_input, current_context):
    try:
        prompt = f"""
        Extract relevant user information from this message to help personalize HR assistance.
        
        Current known information:
        Name: {current_context["name"]}
        Role: {current_context["role"]}
        Department: {current_context["department"]}
        Career Interests: {current_context["career_interests"]}
        Skills to Develop: {', '.join(current_context["skills_to_develop"]) if current_context["skills_to_develop"] else "Not specified"}
        
        User message: "{user_input}"
        
        Extract ONLY new or updated information from the message. Return in JSON format:
        {{"name": "", "role": "", "department": "", "career_interests": "", "skills_to_develop": []}}
        
        If a field is not mentioned, return an empty string or empty list. Only include information explicitly mentioned.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You extract user context information from text. Respond only with the JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        
        extracted_info = eval(response.choices[0].message.content)
        
        # Update current context with new information
        updated_context = current_context.copy()
        for key, value in extracted_info.items():
            if value and value != current_context[key]:
                if key == "skills_to_develop" and value:
                    # Merge skills lists without duplicates
                    updated_context[key] = list(set(current_context[key] + value))
                else:
                    updated_context[key] = value
        
        return updated_context
    except Exception as e:
        logging.error(f"Error extracting user context: {e}")
        return current_context  # Return unchanged context if there's an error

# Get response from OpenAI
def get_completion(prompt, model="gpt-4o-mini"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an HR system focusing on internal employee growth. You compare resumes with job descriptions, provide career advice, match scores, and course recommendations. You also suggest company-specific resources when available."},
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message.content

# Get HR assistant response
def get_response(user_input, context, user_context, conversation_stage):
    try:
        # Determine conversation approach based on stage
        if conversation_stage == "initial" and not user_context["name"]:
            # Initial greeting, focus on gathering basic info
            prompt = f"""
            You are an HR assistant chatbot for Valley Water employees. Your name is Valley.
            This is the beginning of your conversation with a user. Introduce yourself as Valley warmly and ask for their name, role, and department.
            Mention that you can help with HR policies, benefits information, and career development questions.
            Keep your response friendly, concise, and inviting.
            """
        elif "career_development" in conversation_stage:
            # Career development focus
            prompt = f"""
            You are an HR assistant chatbot for Valley Water employees focused on career development.
            
            User Context:
            Name: {user_context["name"]}
            Role: {user_context["role"]}
            Department: {user_context["department"]}
            Career Interests: {user_context["career_interests"]}
            Skills to Develop: {', '.join(user_context["skills_to_develop"]) if user_context["skills_to_develop"] else "Not specified"}
            
            Answer the following question using the provided Valley Water policy information where relevant.
            If policy information isn't relevant to this career question, provide constructive career growth advice.
            Suggest specific learning resources, mentorship opportunities, or development paths when appropriate.
            
            Valley Water Policy Information:
            {context}
            
            Question: {user_input}
            
            Be supportive, practical, and solution-oriented in your response. If you don't have specific information,
            acknowledge that and suggest general best practices or recommend contacting HR directly.
            """
        else:
            # Standard response for policy questions
            prompt = f"""
            You are an HR assistant chatbot for Valley Water employees.
            
            User Context:
            Name: {user_context["name"] if user_context["name"] else "Employee"}
            Role: {user_context["role"] if user_context["role"] else "Valley Water team member"}
            
            Answer the following question using ONLY the provided Valley Water policy information.
            Be clear, helpful, concise, but kind and conversational.
            If the answer cannot be found in the provided information, say "I don't have information on that specific topic in my knowledge base. Please contact HR directly for assistance."
            
            Valley Water Policy Information:
            {context}
            
            Question: {user_input}
            
            If appropriate, mention how this information might relate to their career growth or development.
            """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You are a helpful and conversational HR assistant for Valley Water. Avoid repeating the same phrases across responses. Keep your language varied, concise, and natural. If a point was already made earlier in the conversation, acknowledge it without restating it fully unless necessary for clarity. You're conversational and supportive, finding ways to support employee growth when necessary."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error getting AI response: {e}")
        return "I'm having trouble connecting to my knowledge base right now. Please try again later or contact HR directly."

# Get advice for wellness section
def get_advice(concern, model="gpt-4o-mini"):
    prompt = f"Provide actionable career advice based on the following concern: '{concern}'. Be empathetic and helpful."
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides empathetic and actionable career advice."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

# Determine conversation stage
def determine_conversation_stage(user_input, current_stage):
    user_input_lower = user_input.lower()
    
    career_keywords = ["career", "develop", "growth", "skill", "learn", "training", "advance", "promotion", 
                      "mentor", "opportunity", "certification", "course", "progress", "goal", "improve"]
    
    if any(keyword in user_input_lower for keyword in career_keywords):
        return "career_development"
    elif user_input_lower in ["hi", "hello", "hey"] and current_stage == "initial":
        return "initial"
    else:
        return "policy_question"

# Automatically load PDF from URL
def auto_load_pdf_from_url(url):
    try:
        with st.spinner(f"Loading HR policies from {url}..."):
            # Extract text from PDF URL
            pdf_text = extract_text_from_pdf_url(url)
            if pdf_text:
                # Create vector store
                vector_store = create_vector_store(pdf_text, f"PDF: {url}")
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.knowledge_base_loaded = True
                    logging.info(f"Successfully auto-loaded PDF from {url}")
                    return True
                else:
                    logging.error("Failed to create vector store from PDF")
            else:
                logging.error(f"Failed to extract text from PDF at {url}")
        return False
    except Exception as e:
        logging.error(f"Error in auto_load_pdf_from_url: {e}")
        return False

# Generate TTS audio
def generate_speech(text):
    try:
        response = client.audio.speech.create(
            model="tts-1",  
            voice="nova",   
            input=text
        )
        
        # Get binary audio data
        audio_data = response.content
        
        # Convert to base64 for embedding in HTML
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return audio_base64
    except Exception as e:
        logging.error(f"Error generating speech: {e}")
        return None

# Create HTML with audio element
def get_audio_html(audio_base64):
    audio_html = f"""
    <audio autoplay controls>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html

# Streamlit UI Main Layout
st.title("ValleySpeaks")

# Sidebar for settings and admin features
with st.sidebar:
    st.header("System Settings")
    
    tab1, tab2, tab3 = st.tabs(["Knowledge Base", "Admin Features", "User Profile"])
    
    with tab1:
        # Option to change the default PDF URL
        pdf_url = st.text_input("PDF URL for policies:", value=st.session_state.pdf_url)
        if pdf_url != st.session_state.pdf_url:
            st.session_state.pdf_url = pdf_url
        
        source_type = st.radio("Choose knowledge base source:", ["Auto PDF from URL", "PDF Upload", "Webpage URL"])
        
        if source_type == "Auto PDF from URL":
            if st.button("Load PDF from URL"):
                success = auto_load_pdf_from_url(st.session_state.pdf_url)
                if success:
                    st.success(f"Successfully loaded PDF from URL")
                else:
                    st.error("Failed to load PDF from URL. Please check the URL or try another source.")
        
        elif source_type == "PDF Upload":
            uploaded_file = st.file_uploader("Upload Valley Water HR Policy PDF", type="pdf")
            if uploaded_file and st.button("Load PDF"):
                with st.spinner("Processing PDF..."):
                    # Extract text from PDF
                    pdf_text = extract_text(uploaded_file)
                    if pdf_text:
                        # Create vector store
                        st.session_state.vector_store = create_vector_store(pdf_text, f"PDF: {uploaded_file.name}")
                        st.session_state.knowledge_base_loaded = True
                        st.success(f"Successfully loaded {uploaded_file.name}")
                    else:
                        st.error("Failed to extract text from PDF. Please try another file.")
        
        elif source_type == "Webpage URL":
            url = st.text_input("Enter Valley Water HR Policy webpage URL:")
            if url and st.button("Load Webpage"):
                with st.spinner("Processing webpage..."):
                    # Validate URL
                    try:
                        result = urlparse(url)
                        if all([result.scheme, result.netloc]):
                            # Extract text from webpage
                            web_text = extract_text_from_webpage(url)
                            if web_text:
                                # Create vector store
                                st.session_state.vector_store = create_vector_store(web_text, f"Web: {url}")
                                st.session_state.knowledge_base_loaded = True
                                st.success(f"Successfully loaded {url}")
                            else:
                                st.error("Failed to extract text from webpage. Please try another URL.")
                        else:
                            st.error("Please enter a valid URL including http:// or https://")
                    except:
                        st.error("Please enter a valid URL")
    
    with tab2:
        # Admin Section for Resource Upload
        st.subheader("Upload Learning Resources")
        uploaded_resources = st.file_uploader("Upload Resources (PDF or Text)", type=["pdf", "txt"], accept_multiple_files=True)
        
        # Collect uploaded resource text
        if uploaded_resources:
            resource_texts = []
            for resource in uploaded_resources:
                if resource.type == "application/pdf":
                    resource_texts.append(extract_text(resource))
                elif resource.type == "text/plain":
                    resource_texts.append(resource.read().decode("utf-8"))
            st.session_state.resource_texts = resource_texts
            st.success(f"Successfully loaded {len(resource_texts)} resources")
    
    with tab3:
        # Display user context information
        if st.session_state.user_context["name"]:
            st.subheader("User Profile")
            st.write(f"**Name:** {st.session_state.user_context['name']}")
            if st.session_state.user_context["role"]:
                st.write(f"**Role:** {st.session_state.user_context['role']}")
            if st.session_state.user_context["department"]:
                st.write(f"**Department:** {st.session_state.user_context['department']}")
            if st.session_state.user_context["career_interests"]:
                st.write(f"**Career Interests:** {st.session_state.user_context['career_interests']}")
            if st.session_state and st.session_state.user_context["skills_to_develop"]:
                st.write(f"**Skills to Develop:** {', '.join(st.session_state.user_context['skills_to_develop'])}")
        else:
            st.info("User profile will appear here once you start chatting with the assistant.")
    
    # TTS toggle
    st.header("Text-to-Speech")
    tts_enabled = st.toggle("Enable Voice Responses", value=st.session_state.tts_enabled)
    if tts_enabled != st.session_state.tts_enabled:
        st.session_state.tts_enabled = tts_enabled
    
    st.header("About")
    st.write("This HR Assistant provides information about Valley Water's policies and supports employee career development.")
    st.write("For complex HR matters, please contact the HR department directly.")

# Main area - Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["HR Chat Assistant", "Resume Career Matcher", "Wellness Center"])

with tab1:
    # HR Assistant Chat Interface
    st.header("HR Policy Assistant")
    
    # Check if knowledge base is loaded, if not, attempt auto-load
    if not st.session_state.knowledge_base_loaded and st.session_state.pdf_url:
        auto_load_pdf_from_url(st.session_state.pdf_url)
    
    # Display knowledge base status
    if not st.session_state.knowledge_base_loaded:
        st.warning("No knowledge base loaded. Please load a knowledge base from the sidebar to get accurate answers.")
    else:
        st.write("Ask me about Valley Water's HR policies, benefits, or how I can help with your career development!")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    user_input = st.chat_input("Your question:", disabled=not st.session_state.knowledge_base_loaded)
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Update user context from input
        st.session_state.user_context = extract_user_context(user_input, st.session_state.user_context)
        
        # Determine conversation stage
        st.session_state.conversation_stage = determine_conversation_stage(user_input, st.session_state.conversation_stage)
        
        # Get relevant context
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get relevant context from vector store
                context = retrieve_policy_info(user_input, st.session_state.vector_store)
                
                # Get AI response
                response = get_response(
                    user_input, 
                    context, 
                    st.session_state.user_context, 
                    st.session_state.conversation_stage
                )
                
                # Display the response text
                st.markdown(response)
                
                # Generate and play TTS if enabled
                if st.session_state.tts_enabled:
                    with st.spinner("Generating voice response..."):
                        audio_base64 = generate_speech(response)
                        if audio_base64:
                            st.session_state.last_audio = audio_base64
                            st.markdown(get_audio_html(audio_base64), unsafe_allow_html=True)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    # Career Planning Interface
    st.header("Resume and Job Description Matcher")
    
    uploaded_resume = st.file_uploader("Upload Resume (PDF or Text)", type=["pdf", "txt"], key="resume_uploader")
    uploaded_job_description = st.file_uploader("Upload Desired Job Description (PDF or Text)", type=["pdf", "txt"], key="jd_uploader")
    
    col1, col2 = st.columns(2)
    with col1:
        clear_button = st.button("Clear Analysis")
        if clear_button:
            st.session_state.match_analysis = None
            st.experimental_rerun()
    
    with col2:
        analyze_button = st.button("Analyze Match", disabled=not (uploaded_resume and uploaded_job_description))
    
    if analyze_button and uploaded_resume and uploaded_job_description:
        resume_text = ""
        job_desc_text = ""
        
        if uploaded_resume.type == "application/pdf":
            resume_text = extract_text(uploaded_resume)
        elif uploaded_resume.type == "text/plain":
            resume_text = uploaded_resume.read().decode("utf-8")
        
        if uploaded_job_description.type == "application/pdf":
            job_desc_text = extract_text(uploaded_job_description)
        elif uploaded_job_description.type == "text/plain":
            job_desc_text = uploaded_job_description.read().decode("utf-8")
        
        if resume_text and job_desc_text:
            with st.spinner("Analyzing Resume and Job Description..."):
                # Generate Match Percentage and Analysis
                comparison_prompt = f"""
                Compare the following resume to the desired job description and provide a match score (0-100). 
                Additionally, identify missing skills or qualifications and suggest ways to bridge the gap.

                Resume:
                {resume_text}

                Job Description:
                {job_desc_text}
                """
                st.session_state.match_analysis = get_completion(comparison_prompt)
    
    # Display Match Analysis Results
    if st.session_state.match_analysis:
        st.subheader("Match Percentage and Suggestions:")
        st.write(st.session_state.match_analysis)
        
        # Dynamic Career Development Section
        explore_options = st.checkbox("I would like to explore suggestions for growth.")
        if explore_options:
            career_goal = st.selectbox("What career goal do you want to achieve?", ["Job Performance Improvement", "Career Advancement", "Professional Growth"])
            
            resources_prompt = f"""
            Based on the selected career goal: {career_goal}, and the comparison analysis above, suggest tailored growth plans.

            Include these available resources from the company:
            {''.join(st.session_state.resource_texts) if st.session_state.resource_texts else "No resources uploaded."}
            """
            with st.spinner("Generating personalized growth plan..."):
                analysis = get_completion(resources_prompt)
                st.subheader("Career Development Suggestions:")
                st.write(analysis)
    else:
        st.info("Please upload your resume and desired job description for analysis.")

with tab3:
    # Wellness Center - Breathing Exercise
    st.title("Feeling Overwhelmed?")
    st.title("Take a deep breath or two.")

    st.write("Thinking about the steps to take when it comes to your career is important, BUT it is also important to take care of yourself. The 4-7-8 breathing technique can help you relieve a bit of that stress. Inhale through your nose for 4 seconds, hold for 7 seconds, and exhale out your mouth for 8 seconds.")

    st.subheader("Prepare for Your Breathing Exercise")
    start_button = st.button("Start Countdown to Breathing Exercise")

    if start_button:
        st.write("Starting in...")
        countdown_placeholder = st.empty()
        for seconds in range(3, 0, -1):  
            countdown_placeholder.markdown(f"### {seconds}")
            time.sleep(1)
        countdown_placeholder.markdown("### Go!")
        time.sleep(1)

        st.subheader("Follow the 4-7-8 Breathing Technique")
        stage_placeholder = st.empty()  
        circle_placeholder = st.empty()  

        for _ in range(2):  
            stage_placeholder.markdown("### Inhale...")
            for second in range(4, 0, -1):  
                size = 50 + (4 - second) * 20  
                circle_placeholder.markdown(f"""
                <div style="width:{size}px; height:{size}px; background-color:lightblue; border-radius:50%; margin:auto; display:flex; align-items:center; justify-content:center;">
                    <p style="font-size:20px; font-weight:bold; color:black;">{second}</p>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1)

            stage_placeholder.markdown("### Hold...")
            for second in range(7, 0, -1):  
                size = 130 
                circle_placeholder.markdown(f"""
                <div style="width:{size}px; height:{size}px; background-color:lightblue; border-radius:50%; margin:auto; display:flex; align-items:center; justify-content:center;">
                    <p style="font-size:20px; font-weight:bold; color:black;">{second}</p>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1)

            stage_placeholder.markdown("### Exhale...")
            for second in range(8, 0, -1):  
                size = 130 - (8 - second) * 10  
                circle_placeholder.markdown(f"""
                <div style="width:{size}px; height:{size}px; background-color:lightblue; border-radius:50%; margin:auto; display:flex; align-items:center; justify-content:center;">
                    <p style="font-size:20px; font-weight:bold; color:black;">{second}</p>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1)

        st.success("Breathing exercise complete! Feel free to share how you're feeling.")

    with st.container():
        st.subheader("How Are You Feeling?")
        concern = st.text_area("Share your job-related concerns below, and I'll provide some helpful advice:")

        if st.button("Submit Concern"):
            if concern:
                advice = get_advice(concern)
                st.subheader("Your Personalized Career Advice:")
                st.write(advice)
            else:
                st.warning("Please share your concern to receive advice.")