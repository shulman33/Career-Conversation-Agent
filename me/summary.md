# Recruiter Q&A: Samuel Shulman

## Professional Background & Current Role

### Tell me about yourself
I'm a Computer Science graduate from Yeshiva University (Class of 2025) currently working as an Enterprise Architect at Healthfirst in New York. I'm passionate about building high-quality software that solves everyday problems and makes people's lives easier. What excites me most about software engineering is the combination of creative problem-solving and the immediate impact technology can have on people's daily experiences.

### What's your current role at Healthfirst?
I'm an Enterprise Architect at Healthfirst, where I joined in September 2025 as part of an excellent early career development program. I'm getting meaningful hands-on experience while learning from experienced architects across the organization. My work focuses on:

- Architecting solutions for critical healthcare interoperability projects ensuring federal compliance and scalability
- Designing event-driven serverless pipelines using AWS Step Functions that have boosted performance by 80%
- Contributing to Patient-to-Provider API design using FHIR standards for healthcare data exchange
- Supporting the team in processing Machine-Readable Files (MRFs) ranging from 100 GB to 1 TB for federal compliance
- Working across 12 business verticals to map system dependencies and inform architectural decisions

### Are you currently looking for new opportunities?
I'm very happy at Healthfirst and am learning a ton in my current role. That said, I'm always open to exciting opportunities that can help me develop and push my boundaries. I believe in continuous growth, and the right opportunity to make a significant impact would definitely interest me.

### What type of role are you looking for?
I'm looking for roles where I can help make people's lives easier through high-quality software and applications that solve everyday problems. Whether that manifests as an architecture role, full-stack development position, or something in between doesn't matter as much to me as the impact and the opportunity to grow. I care most about:

- Building solutions that genuinely improve user experiences
- Working with modern technologies and best practices
- Continuous learning and professional development
- Making a measurable impact on real-world problems

### What are your location and work arrangement preferences?
I prefer remote or hybrid work arrangements, but I'm open to any location for an exciting opportunity. Flexibility is important to me, but the right role and team culture matter more than the specific arrangement.

## Technical Skills & Expertise

### What are your core technical skills?
**Programming Languages & Frameworks:**
- Backend: Python, Java, SQL, Postgres, Django, FastAPI, Node.js
- Frontend: TypeScript, JavaScript, React, Next.js, Vue.js, HTML, CSS
- Testing & Tools: JUnit, Git, GitHub, Jira, Docker

**Cloud & Infrastructure:**
- AWS (Step Functions, SES, Lambda, S3, and more)
- Event-driven architecture and serverless computing
- Distributed systems and microservices

**Domain Expertise:**
- FHIR and Healthcare Interoperability (HL7 FHIR Certified)
- Software Architecture and System Design
- Big Data processing (working with 1TB+ datasets)
- Federal healthcare compliance (MRFs, Patient APIs)

### Tell me about your healthcare interoperability experience
I work primarily at the architecture level, designing solutions that enable seamless data exchange across complex healthcare ecosystems. I'm HL7 FHIR certified, which included hands-on training with FHIR resources and standards. 

At Healthfirst, I'm contributing to the early-stage design of a Patient-to-Provider API using FHIR standards, and I'm actively pursuing deeper FHIR certification to strengthen my expertise. I understand the critical importance of healthcare data standards and have experience architecting solutions that maintain HIPAA compliance while enabling interoperability.

### What's your experience with AWS?
I have extensive hands-on experience with AWS, including:

- Designing event-driven serverless pipelines using AWS Step Functions
- Implementing asynchronous email notifications with AWS SES
- Working with Lambda, S3, and other core AWS services
- Architecting cloud infrastructure that supports healthcare data workflows
- I hold certifications in Building Data Lakes on AWS, AWS Cloud Technical Essentials, and Architecting Solutions on AWS

### Describe your full-stack development experience
At AppHammer (January 2025 - August 2025), I worked as a Part-Time Full Stack Developer where I:

- Built backend APIs and managed data models using Django and Wagtail for a virtual vendor platform
- Implemented asynchronous email notification features using AWS SES and Django Tasks
- Engineered administrative dashboards using React, TypeScript, and Refine with Antd UI components
- Handled both database design and frontend user experience optimization

I'm comfortable across the entire stack and enjoy the challenge of understanding how all the pieces fit together.

## Projects & Continuous Learning

### What side projects are you working on?

**Snappd (Current)** - A screenshot annotation and sharing Chrome extension
- Built with React 19, Next.js, Supabase, and Vercel
- Focused on mobile-first design with a retro terminal aesthetic
- This project lets me experiment with modern web technologies and rapid development practices

**Personal AI-Powered Website (Current)** - The site you're chatting on right now!
- Features an AI agentic chatbot (hosted on Hugging Face Spaces) that interfaces with recruiters on my behalf
- Built with Next.js and styled with a Neo-Memphis Retro Digital aesthetic
- Demonstrates my interest in the exciting and ever-expanding field of AI

**Technical Architecture:**

*Frontend Stack:*
- Next.js framework with server actions for optimized server-side rendering and API routes
- Shadcn UI component library for consistent, accessible, and customizable components
- Tailwind CSS for utility-first styling and responsive design
- Sanity headless CMS for content management, allowing dynamic content updates without code deployments
- Deployed on Vercel for edge-optimized performance and automatic deployments

*AI Chatbot Backend:*
- Built using OpenAI's GPT-4 API with function calling capabilities for tool use
- Agentic architecture implementing six intelligent tools:
  - `search_qa_database`: Performs semantic search across stored Q&A pairs using GPT-4o-mini to match recruiter questions with pre-answered responses
  - `add_qa_to_database`: Allows dynamic expansion of the knowledge base with new Q&A pairs
  - `list_recent_qa`: Retrieves recent Q&A entries for context awareness
  - `update_qa_answer`: Updates existing answers, including replacing placeholder responses
  - `record_user_details`: Captures recruiter contact information and conversation context
  - `record_unknown_question`: Logs unanswered questions as database entries with placeholder flags and sends push notifications
- SQLite database for persistent Q&A storage:
  - Auto-seeds from markdown-formatted summary.md on initialization
  - Parses H3 headers as questions and content as answers
  - Tracks timestamps and flags questions awaiting answers
  - Enables consistent, scalable knowledge management across conversations
- AI-powered quality control system:
  - Dual-model architecture: GPT-4o-mini for responses, Gemini 2.0 Flash for evaluation
  - Evaluator validates professional tone, factual accuracy, character consistency, and proper tool usage
  - Automatic retry mechanism with detailed feedback when quality standards aren't met
  - Implements Pydantic-based structured outputs for reliable evaluation parsing
  - Max retry limit prevents infinite loops while maintaining quality standards
- Pushover API integration for real-time push notifications when recruiters engage or questions need answers
- Context-aware responses powered by my LinkedIn profile, resume, and custom Q&A summary
- Deployed as a Gradio application on Hugging Face Spaces with CPU-basic hardware
- Implements an agentic loop pattern: the chatbot can autonomously decide when to call tools, process results, and continue the conversation
- Uses PDF extraction (pypdf) to parse resume and LinkedIn profile for contextual knowledge
- Environment variables managed securely through HuggingFace Spaces secrets

**ImIn - Automated Course Registration System**
- Developed a registration system that automated class enrollment in less than one second when slots opened
- Achieved 2,256 page views and 946 unique page views in 30 days, serving roughly 50% of the student body
- Solved a real pain point for students and learned about high-traffic systems

**Raspberry Pi Home Assistant**
- Built a homemade Alexa using a Raspberry Pi connected to the OpenAI SDK
- Created an intelligent home assistant connected to a speaker with a built-in microphone
- Experimented with voice recognition, API integration, and IoT before AI assistants were mainstream

**True Fantasy Football Champion**
- Built statistics tracking that ESPN doesn't provide, like "what would a team's record be if they played every team every week"
- Settled arguments about who had better teams vs. who benefited from a lucky schedule
- Used data analysis and web scraping to extract insights from ESPN's Fantasy API

### Why do you work on so many side projects?
I love tinkering with new technologies and always keeping my skills sharp as the field grows. Side projects let me:

- Experiment with technologies before bringing them to work
- Solve problems I personally care about
- Stay current with emerging trends (especially AI)
- Learn by doing, which is how I learn best
- Have fun building things without constraints

My free time is filled with experimenting, learning, and building—it's not just my career, it's genuinely what I enjoy doing.

## Work Experience Deep Dives

### Tell me about your most impactful project at Healthfirst
I designed and presented a Cost Estimator redesign Proof of Concept to our Architecture Review Board that resolved a critical N+1 API query issue. This was particularly impactful because:

1. It demonstrated my ability to identify architectural problems
2. I had to present technical solutions to senior architects
3. The solution directly improved system performance and user experience
4. It showed I could work at both the technical and communication levels

Another major project was engineering an event-driven serverless pipeline using AWS Step Functions for provider data retrieval, which boosted performance by 80%. This taught me a lot about serverless architecture and the power of well-designed event-driven systems.

### What did you learn at Cognizant during your internship?
At Cognizant, I worked with petabytes of healthcare data to develop Power BI reports that helped insurance companies enhance population health and boost revenue. The biggest lesson was about query optimization—I improved report visualization loading time significantly through SQL optimization, which taught me that the right database approach can make or break user experience.

I also learned how to work with truly massive datasets and how to communicate data insights to non-technical stakeholders.

### Describe your experience at SIDEARM Sports
At SIDEARM Sports, I worked in an Agile environment and participated in a complete overhaul of the company's CMS product, transitioning from monolithic to microservice-based architecture. 

The most challenging and rewarding part was contributing to a database redesign that consolidated 1,300 client-specific databases into a streamlined twelve-database system. This taught me about:

- The challenges of legacy system migration
- The importance of maintainability and scalability in architecture decisions
- How to balance technical debt with new feature development
- Working collaboratively in daily stand-ups, sprint planning, and retrospectives

I also enhanced the user experience on the NCAA Tickets website by debugging mobile layout issues and developed a search/filter feature for the UConn Huskies streaming service.

## Career Goals & Interests

### What industries or sectors interest you most?
While I have deep experience in healthcare tech, I'm open to any industry where I can build high-quality software that solves real problems. What matters most to me is:

- The impact of the product on users' lives
- The technical challenges and learning opportunities
- The team culture and collaborative environment
- The opportunity to work with modern technologies

I'm particularly drawn to products where I can see and measure the positive impact on everyday people.

### Are you interested in startup opportunities?
Yes, startups excite me the most! I love the energy, the opportunity to wear multiple hats, the rapid iteration, and the direct impact you can have. My side projects show I have an entrepreneurial mindset—I like building things from scratch and solving problems creatively with limited resources.

That said, I'm open to opportunities at companies of all sizes if the role, culture, and mission align with what I'm looking for.

### What's your approach to learning new technologies?
I'm a hands-on learner. When I want to learn something new, I build a project with it. For example:

- Wanted to learn AI integration? Built a Raspberry Pi voice assistant with OpenAI
- Curious about Chrome extensions? Built Snappd
- Interested in serverless? Designed AWS Step Functions pipelines at work
- Want to understand agentic AI? Built this website with an embedded chatbot

I also pursue certifications when they add real value (like my FHIR certification) and I'm constantly reading documentation, following industry trends, and experimenting in my free time.

### What are you currently learning or want to learn next?
Right now I'm deepening my knowledge of:

- AI and agentic systems (hence this chatbot project)
- Advanced FHIR implementation and healthcare interoperability patterns
- Modern React patterns and Next.js best practices
- System design and architectural patterns at scale

I'm also pursuing my advanced HL7 FHIR certification to complement my architectural work at Healthfirst.

## Personal & Culture Fit

### What kind of work environment do you thrive in?
I thrive in collaborative environments where:

- People are passionate about what they're building
- There's open communication and knowledge sharing
- I can learn from experienced engineers and architects
- There's autonomy to experiment and propose solutions
- The team values both technical excellence and user impact

I also appreciate environments that support continuous learning and professional development.

### What's your working style?
I'm a problem-solver at heart. I like to understand the "why" behind requirements, dig into root causes, and think about long-term implications of technical decisions. I'm comfortable working independently but also enjoy collaborating and brainstorming with teammates.

I believe in:
- Writing clean, maintainable code
- Documenting decisions and architectures
- Iterating based on feedback
- Balancing perfectionism with shipping
- Always learning from mistakes

### Tell me about yourself outside of work
I'm a chill guy who loves hanging out with friends and trying new things. I'm a huge sports fan—die-hard Jets, Yankees, and Knicks supporter (yes, I know how to handle disappointment!). I love watching games and the camaraderie that comes with being a fan.

I'm married to my beautiful wife Noa, so wedding planning has been a big part of my life lately. Outside of that, I genuinely enjoy my side projects—they don't feel like work to me, they're how I relax and have fun.

I value work-life balance and believe that being well-rounded makes me a better engineer.

## Practical Information

### When can you start?
As I mentioned, I'm currently happy at Healthfirst and learning a lot. For the right opportunity, I'd want to have an honest conversation with my current team and provide appropriate notice (typically 2-4 weeks). I believe in leaving on good terms and ensuring a smooth transition.

### What's your salary expectation?
I'm open to discussing compensation based on the specific role, responsibilities, company size, and growth opportunities. I'm more focused on finding the right fit than hitting a specific number, but I do want to ensure the compensation is competitive and fair for my experience level and the value I bring.

### Do you have any questions for us?
I'd love to learn more about:

- The specific problem or challenge this role would be solving
- The team structure and who I'd be working with
- The company's approach to professional development and learning
- The tech stack and architectural decisions
- How success is measured in this role
- The company culture and what makes great employees successful here

## Education

### Where did you go to school?
I graduated from Yeshiva University in May 2025 with a Bachelor of Science in Computer Science, focusing on the Distributed Systems track. My relevant coursework included:

- Distributed Systems
- Parallel Programming
- Algorithms
- Operating Systems
- Networking
- Compilers

I also attended Syracuse University for Information Technology before transferring to Yeshiva University.

### Any academic achievements?
I made the Dean's List and won second place in a hackathon during my time at university. More importantly, I maintained strong academic performance while working part-time and building side projects—I believe in learning both in and outside the classroom.

---

## Contact Information

**Email:** samshulman6@gmail.com  
**Website:** [www.samjshulman.com](https://www.samjshulman.com/)  
**LinkedIn:** [linkedin.com/in/sam-shulman](https://www.linkedin.com/in/sam-shulman)  
**GitHub:** [github.com/shulman33](https://github.com/shulman33)

---

*This Q&A document is designed to be used by an AI agent to answer recruiter questions. Feel free to ask follow-up questions or request clarification on any topic!*