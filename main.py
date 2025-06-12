from fastapi import FastAPI,WebSocket,WebSocketDisconnect,File,UploadFile,Form
import openai
import os
from dotenv import load_dotenv
import pyttsx3
from fastapi.middleware.cors import CORSMiddleware
import cv2
import dlib
from scipy.spatial import distance
from deepface import DeepFace
import PyPDF2
import io
import asyncio
from pydantic import BaseModel
import base64
import os
from PIL import Image
import pdf2image
import io
import google.generativeai as genai
import base64
import shutil
import numpy as np
import requests

load_dotenv()

openai.api_key= os.getenv("OPENAI_KEY")
app = FastAPI()

os.makedirs("uploads", exist_ok=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

class ResumeData(BaseModel):
    resume_dt : str
    job_description : str
    jobType :str
    role : str
    experience : int
    interview_type : str

jobDescription = ""
resume = ""
job_type = ""
role = ""
experience = 0 
interviewType = ""

@app.post("/resume")
async def get_resume(resumedt : ResumeData):    
    global resume      
    global jobDescription
    global experience
    global  role
    global job_type
    global interviewType

    resume = resumedt.resume_dt
    jobDescription = resumedt.job_description
    job_type = resumedt.jobType
    role = resumedt.role
    experience = resumedt.experience
    interviewType = resumedt.interview_type
    

interation = []

def generate_question(role):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Ensure you're using the correct model
        messages=[
            {"role": "system", "content": f"""You are an interviewer for the role of {role}{job_type}.He is having {experience} years of experience.Generate the question to ask to the candidate. 
                The previous questions and candidate responses are : {interation}.
                - You have to generate only one question to aks.
                - Analyse previous interaction and generate appropriate question.
                - NOTE THAT INTERVIEW SHOULD BE {interviewType}. SO ONLY {interviewType} QUESTIONS SHOULD BE PRESENT.
                - Maintaion the flow of question answering.
                - *Your first question should be "Tell me about yourself."*
                - If you are going to ask first question (i.e. previous responses are null), it should be "Tell me about yourself"
                - Do not stick with the same flow but ask questions which covers entire aspects of interview.
                - Below i have give the resume of the candidate, analyze it and generate the proper set of questions
                - Add technical and Non-Technical Questions also
                - Add technical questions also for perticular role
            Note : Generate only a question. Does not include any introduction.
                **Candidate resume ** :{resume} 
                **Job descriptions** : {jobDescription}

                -- MAKE SURE YOU ARE GENERATING **BOTH** TECHNICAL AS WELL AS NON-TECHICAL QUESTIONS ALSO.
                -- SPECIAL NOTE : YOU HAVE TO REFERE THE PREIOUS RESPONSE ALSO IF REQUIRED.

            Note : Keep questions alligned with uploaded resume and job descriptions.
             """},            
        ]
    )
    return response.choices[0].message['content'].strip()


def analyze_response(question, answer):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Ensure you're using the correct modelsss
        messages=[
            {"role": "system", "content": "You are an expert interview response analyzer."},
            {"role" : "user","content" : f"""You have given the question and it's corresponding response by candidate
                Question : {question}
                Response : {answer}
                Generate best feedback to be given to the candidate
                - Ana
                - If candidate ask you to to answer the previuetion you have aske, give it's ideal response.
                - Give ideal response only when explicitally asked by the candidate.
                - Generate feedback in paragraph format
                - Generate short but swwet feedback 
                - Do not add any introduction
                - Go through Candidate details and job descriptions and alighn your response to them

                **Candidate resume ** :{resume} 
                **Job descriptions** : {jobDescription}

                
            
            **If you think, give ideal response of the question also.Do not say that 'I can provide you ideal response or similar' . If you think, directly state**
            -- give ideal response in same paragraph to maintantain the overrall flow of interactions.
                """}
        ]
    )
    return response.choices[0].message['content'].strip()  # Access the correct field


@app.post("/clearData")
def clearData():
    global resume      
    global jobDescription
    global experience
    global  role
    global job_type
    global interviewType
    global interation

    jobDescription = ""
    resume = ""
    job_type = ""
    role = ""
    experience = 0 
    interviewType = ""
    interation = []

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

async def face_analysis(data):    
    try:
        # Convert received binary image data to OpenCV format
        frame = Image.open(io.BytesIO(data))
        frame_rgb = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

        # Debug: Save received frame to verify it contains a face
        cv2.imwrite("received_frame.jpg", frame_rgb)

        # Detect faces
        faces = detector(gray)

        print(f"Detected {len(faces)} faces")  # Debugging

        if not faces:
            return {"emotion": "No Face Detected", "eye_contact": False}

        # Run DeepFace emotion analysis asynchronously
        emotion_analysis = await asyncio.to_thread(
            DeepFace.analyze, frame_rgb, actions=["emotion"], enforce_detection=False, detector_backend="opencv"
        )
        emotion = emotion_analysis[0]['dominant_emotion']

        # Check for eye contact
        eye_contact = False
        for face in faces:
            landmarks = predictor(gray, face)

            # Extract left and right eye coordinates
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            # Calculate EAR (Eye Aspect Ratio)
            left_eye_ear = eye_aspect_ratio(left_eye)
            right_eye_ear = eye_aspect_ratio(right_eye)
            ear = (left_eye_ear + right_eye_ear) / 2.0

            print(f"EAR: {ear}")  # Debugging

            # Determine eye contact (adjust EAR threshold if needed)
            eye_contact = ear > 0.2  

        return {"emotion": emotion, "eye_contact": int(eye_contact)}

    except Exception as e:
        print(f"Error processing frame: {e}")
        return {"emotion": "Error", "eye_contact": False}

@app.websocket("/video")
async def videoSocket(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_bytes()
        result = await face_analysis(data)  # Await the coroutine
        await websocket.send_json(result)   # Send the result

@app.websocket("/interview")
async def interview(websocket: WebSocket):
        await websocket.accept()
        print(jobDescription)
        try:
        # await websocket.send_text(f"welcome to the interview preparation for {role} role!")
            while True:
                question = generate_question(role)
                await websocket.send_text(question)         

                answer = await websocket.receive_text()                
                interation.append({"question": question, "answer": answer})
                feedback = analyze_response(question, answer)
                await websocket.send_text(feedback)
            
        except WebSocketDisconnect:
            print(f"Client disconnected from the interview for role: {role}")



###### ATS System
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def getGeminiResponse(input_prompt,pdf_content,job_description):
    model=genai.GenerativeModel('gemini-2.0-flash')
    response=model.generate_content([input_prompt]+pdf_content+[job_description])
    return response.text

def input_pdf_setup(pdf_path):
    with open(pdf_path,"rb") as file:
        images = pdf2image.convert_from_bytes(file.read())

    page_parts = []

    for image in images:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr,format = "JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        page_parts.append(
            {
            "mime_type" : "image/jpeg",
            "data":base64.b64encode(img_byte_arr).decode()
            }
        )
    os.remove(pdf_path)
    return page_parts

input_prompt1 = """
    You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
    Please share your professional evaluation on whether the candidate's profile aligns with the role. 
    Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
    Make sure that do **NOT** include any heading or introduction.

    Give proper views on the resume.
     - Do not include initial introduction but return actual data only.
     - Use proper markdowns, linespacings and highlighrts for better visualisation.
     - DO NOT ADD ANY INITIAL INTRODUCTION,DIRECTLY JUMP ON POINT.
     - Generate Well structured and detailed response.
"""




input_prompt3 = """
    You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
    your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
    the job description. First the output should come as percentage and then keywords missing and last final thoughts.
     - Do not include initial introduction but return actual data only.
     - Use proper markdowns, linespacings and highlighrts for better visualisation.
     - DO NOT ADD ANY INITIAL INTRODUCTION,DIRECTLY JUMP ON POINT.
     - Generate detailed and well structured response.

 """

page_parts =""
atsResume = ""
ATSJobdescription = ""
atsPrompt = ""


UPLOAD_DIR = "uploads"

@app.post("/get_resume_file")
async def get_resume(ATSdescription: str = Form(),prompt_number : int = Form(),file : UploadFile = File(...)):
    global page_parts, ATSJobdescription , atsPrompt , input_prompt1 , input_prompt3

    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    page_parts = input_pdf_setup(file_path)
    ATSJobdescription = ATSdescription   
    
    if prompt_number == 1:
        atsPrompt = input_prompt1
    elif prompt_number == 2:
        atsPrompt = input_prompt3

   
@app.post("/ats_response")
async def sendAtsData():
    result =  getGeminiResponse(atsPrompt,page_parts,ATSJobdescription)
    return result

@app.post("/jobSearch")
def getJobs(resume : UploadFile = File(...)):
    prompt = """
        You are the skilled resume analyzer. Your task is to analyze the gievn resume and identify the job roles which are matching with them.
        Your resonse will be used to search for the job roles for the user. 
        Analyze carefully and list down the output in following format. 

        output_format : Jobs for <role1 , role2 , .....> Roles in India.

        NOTES : 
            DO NOT ADD ANY HEADING OR INTRODUCTION, DIRECLY GIVE OUTPUT.

"""
    file_path = f"{UPLOAD_DIR}/{resume.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)
    
    page_parts = input_pdf_setup(file_path)
    model=genai.GenerativeModel('gemini-2.0-flash')
    
    response=model.generate_content([prompt]+page_parts)
    print(response.text)

    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": "88bd258fa3mshad2e265813778ccp1f6954jsn2063843372ac",
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    params = {"query": f"{response.text}", "page": "1"}
    jobs = requests.get(url, headers=headers, params=params)

    jobs_json= jobs.json()
    print(jobs_json["data"])
    return {"jobs" : jobs_json.get("data")}
import uvicorn

if __name__ == "__main__":
    uvicorn.run("your_filename:app", host="0.0.0.0", port=8080)
