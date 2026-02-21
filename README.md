**NoOneLeftAlone.exe**



Intelligent Real-Time Crowd Monitoring \& Alert System



NoOneLeftAlone.exe is an AI-powered real-time monitoring system designed to enhance safety and situational awareness in public and semi-public environments.



The system leverages Computer Vision and Machine Learning to detect human presence from live video feeds, monitor crowd density, and automatically generate alerts when predefined safety thresholds are exceeded.



It integrates hardware (camera devices) with a software-based monitoring dashboard to provide a scalable and proactive crowd management solution.



**1. Problem Statement**



Overcrowding and lack of real-time monitoring in public spaces such as educational institutions, transportation hubs, events, and commercial areas can lead to safety risks.



Manual monitoring methods are:



* Inefficient



* Delayed



* Error-prone



* There is a need for an automated, intelligent system that can:



* Detect human presence in real time



* Evaluate crowd density



* Trigger alerts proactively



* Maintain historical data records



**2. Proposed Solution**



NoOneLeftAlone.exe provides an automated monitoring pipeline:



* Capture live video input



* Detect humans using AI-based object detection



* Count detected individuals



* Compare count against configured threshold



* Generate alert if threshold is exceeded



* Store event details in database



* Display monitoring data via web dashboard



* This enables early detection of overcrowding and improves response time.



**3. System Architecture**



Camera → OpenCV Processing → YOLO Detection → Crowd Counter → Threshold Evaluation → Alert Generation → Database → Web Dashboard



**4. Technology Stack**



* Programming Languages



* Python



* HTML



* CSS



* JavaScript



* Frameworks \& Libraries



* OpenCV – Image processing and frame handling



* YOLO – Real-time object detection



* Flask – Backend web framework



* Database



* MySQL / SQLite



* Hardware Requirements



* Laptop / PC



* Webcam / CCTV Camera



* Minimum 8GB RAM recommended



**5. Core Modules**



* Admin Module



* Configure threshold values



* Manage crowd records



* Monitor alerts



* Event Manager Module



* Monitor camera feeds



* Review alert history



* Track system activity



* Camera Module



* Captures real-time video stream



* Crowd Record Entity



* Stores detected human count



* Maintains timestamp



* Alert Entity (Weak Entity)



* Triggered when threshold condition is satisfied



* Dependent on Crowd Record



**6. Key Features**



* Real-time human detection



* Threshold-based alert generation



* Database logging of events



* Administrative dashboard



* Hardware and software integration



* Scalable system architecture



**7. Installation \& Setup**



Step 1 – Clone Repository


```Bash
git clone https://github.com/your-username/NoOneLeftAlone.exe.git

cd NoOneLeftAlone.exe

```

Step 2 – Install Dependencies


``` Bash
pip install -r requirements.txt

```

Step 3 – Run Application


```Bash
python app.py

```

Access the application via:


```Code
http://127.0.0.1:5000

```

**8. Future Enhancements**



* Multi-camera integration



* SMS / Email alert notifications



* Cloud deployment



* Emotion detection extension



* Mobile dashboard application



**9. Project Context**



This project was developed as part of a hackathon initiative focused on intelligent automation, public safety, and AI-driven monitoring systems.



**10. Team**



Devi Parvathy K

Sethulakshmi P R



Hackathon: Think-Her-Hack































