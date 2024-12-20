import random
wording_styles = [
    "Use clear and simple language to explain the task.",
    "Make the instructions more fun by using playful language.",
    "Use positive words to motivate the reader.",
    "Keep the language casual to make the task feel approachable.",
    "Use familiar words that your audience will recognize.",
    "Turn complex terms into easy-to-understand phrases.",
    "Use friendly and supportive language throughout.",
    "Ask direct questions to engage the reader’s curiosity.",
    "Incorporate humor where appropriate to keep it light.",
    "Use inclusive language to make everyone feel involved.",
    "Keep sentences short and snappy to maintain interest.",
    "Use imaginative language to spark creativity in the reader.",
    "Encourage action by using enthusiastic and energetic words.",
    "Add a conversational tone to make the reader feel guided.",
    "Use storytelling to explain parts of the task when possible.",
    "Avoid using formal or stiff language that may feel boring.",
    "Use encouraging phrases like ‘You’ve got this!’ or ‘Keep going!’",
    "Incorporate examples that are relevant and fun for the reader.",
    "Keep the language upbeat to make the task feel exciting.",
    "Avoid overly serious or strict wording.",
    "Use vivid descriptions to paint a picture of the task.",
    "Create a sense of adventure with dynamic language.",
    "Use verbs that inspire action like ‘Create’, ‘Build’, or ‘Explore’.",
    "Try using metaphors or comparisons to simplify instructions.",
    "Use second person (you) to speak directly to the reader.",
    "Highlight the fun parts of the task in the wording.",
    "Reassure the reader that they can succeed with positive language.",
    "Add light, playful commands like ‘Let’s get started!’.",
    "Make steps feel achievable by using words like ‘easy’ and ‘simple’.",
    "Show excitement in the task by using exclamation marks when appropriate.",
    "Use friendly language that sounds like you’re speaking to a friend.",
    "Remove any words that could feel intimidating or confusing.",
    "Phrase instructions as invitations, like ‘Why not try...?’",
    "Use examples that connect to the reader’s everyday life.",
    "Simplify instructions by using plain language and avoiding filler words.",
    "Keep the tone encouraging, like a helpful guide.",
    "Use direct language to give clarity on what’s expected.",
    "Engage curiosity by using language that invites exploration.",
    "Use playful verbs like ‘discover’, ‘imagine’, or ‘solve’.",
    "Build excitement with words like ‘awesome’, ‘exciting’, or ‘cool’.",
    "Avoid technical terms and replace them with relatable phrases.",
    "Let the reader know they’re doing great with phrases like ‘Nice work!’.",
    "Create a sense of fun challenge with words like ‘Can you?’ or ‘How about trying?’",
    "Use active voice to keep the language direct and motivating.",
    "Paint a mental image with descriptive verbs like ‘design’ or ‘craft’.",
    "Break down tricky concepts into fun, digestible parts.",
    "Encourage curiosity with open-ended language like ‘What if you...?’",
    "Make the task feel exciting by using upbeat adjectives.",
    "End with an encouraging phrase like ‘You’re almost there!’.",
    "Use language that sparks imagination and excitement for the task."
]

solver_personas = [
    "You are a detail-oriented student who enjoys breaking down complex problems into manageable steps.",
    "You are quick to spot patterns in math problems and enjoy coming up with creative solutions.",
    "You excel at collaborating with others, often leading group discussions to solve challenging problems.",
    "You approach each math problem methodically, always double-checking your work for accuracy.",
    "You are known for your curiosity, often asking thought-provoking questions that deepen class discussions.",
    "You love applying math concepts to real-world situations and are always thinking practically.",
    "You are highly organized, keeping neat and detailed notes that help you solve problems efficiently.",
    "You are enthusiastic about problem-solving and love experimenting with different strategies to find solutions.",
    "You have a natural talent for simplifying complex word problems and explaining your reasoning to peers.",
    "You are always eager to help others in class, often offering helpful insights during group work.",
    "You enjoy thinking outside the box, frequently coming up with unconventional but effective solutions.",
    "You are a competitive problem solver who thrives in timed challenges and math competitions.",
    "You are a calm and patient learner, often spending extra time to ensure you fully understand each problem.",
    "You are highly analytical, breaking down word problems into logical steps with precision.",
    "You are a visual learner who excels at drawing diagrams to represent and solve math word problems.",
    "You are quick to grasp abstract concepts and enjoy exploring advanced mathematical theories on your own.",
    "You are a perfectionist, always striving for the most accurate and complete solution to every problem.",
    "You love puzzles and see math problems as an opportunity to engage your strategic thinking skills.",
    "You are a strong communicator who excels at explaining your problem-solving process to the class.",
    "You are very competitive and enjoy challenging yourself with the hardest problems in class.",
    "You are highly imaginative, often creating stories around math problems to make them more engaging.",
    "You are a fast learner who quickly adapts to new problem-solving techniques introduced in class.",
    "You are meticulous and enjoy finding the most efficient way to solve math word problems.",
    "You are enthusiastic and bring a positive energy to class, making even the toughest problems fun.",
    "You are a quiet but confident problem solver who consistently gets high marks on every assignment.",
    "You enjoy the challenge of multi-step problems, seeing them as puzzles to be solved piece by piece.",
    "You love numbers and are always excited to discover new ways to manipulate them in word problems.",
    "You are always thinking critically, often questioning the assumptions behind word problems to explore alternative solutions.",
    "You are intuitive with numbers and often solve problems mentally faster than your peers.",
    "You are curious about the practical applications of math and enjoy discussing how math affects everyday life.",
    "You are a careful and reflective thinker, always taking time to fully understand problems before diving into solutions.",
    "You are a competitive and focused student who enjoys pushing yourself with increasingly difficult problems.",
    "You are highly adaptive, easily shifting your approach when you encounter a challenging problem.",
    "You are a confident student who takes pride in solving complex problems without needing much help from others.",
    "You have a vivid imagination and use creative approaches to understand and solve math word problems.",
    "You are a precise and methodical thinker who enjoys solving problems in a structured way.",
    "You are patient and persistent, never giving up until you have found the right solution.",
    "You are logical and organized, excelling at problems that require detailed, step-by-step solutions.",
    "You love exploring different problem-solving techniques and often introduce new methods to the class.",
    "You are a collaborative learner who thrives in group problem-solving activities.",
    "You enjoy using real-life examples to make math problems more relatable and easier to solve.",
    "You are a fast problem solver who enjoys figuring out shortcuts and efficient methods.",
    "You are a critical thinker who always considers multiple angles before deciding on the best approach.",
    "You are an empathetic student who enjoys helping your peers understand difficult math problems.",
    "You enjoy exploring math theory and often bring up interesting questions about why certain methods work.",
    "You are a creative thinker who enjoys finding innovative ways to solve classic math word problems.",
    "You are a self-motivated student who takes pride in mastering difficult concepts independently.",
    "You are a determined learner who always stays focused, even when problems seem overwhelming.",
    "You enjoy using technology to solve math problems and frequently incorporate coding into your solutions.",
    "You are a big-picture thinker who enjoys solving complex problems with multiple parts."
]

manager_personas = [
    "You are the go-to person for streamlining complex processes into simple, actionable steps.",
    "You excel at breaking down data analysis into digestible insights for your team.",
    "You are a master at motivating others through your energetic leadership and clear vision.",
    "You bring out the best in others by fostering collaboration and open communication.",
    "You know exactly how to delegate tasks efficiently, ensuring nothing falls through the cracks.",
    "You have a unique talent for guiding others through intricate technical challenges.",
    "You are an expert at coaching teams to achieve tight deadlines without compromising quality.",
    "You always create a learning environment where everyone feels comfortable asking questions.",
    "You thrive at helping coworkers understand financial reports and budget planning.",
    "You are a natural mentor, turning everyday tasks into growth opportunities for others.",
    "You make onboarding new employees seamless by providing clear instructions and support.",
    "You know how to resolve conflicts with diplomacy and ensure that everyone feels heard.",
    "You have a gift for translating high-level strategies into practical, day-to-day actions.",
    "You consistently inspire your team to exceed expectations with your positive reinforcement.",
    "You lead by example, showing your team how to approach challenges with creativity and logic.",
    "You are a problem-solver who always finds innovative solutions to operational issues.",
    "You excel at making data-driven decisions that guide your team toward success.",
    "You are skilled at coaching others to prioritize tasks and manage their time effectively.",
    "You always maintain a calm and organized environment, even in high-pressure situations.",
    "You know how to balance giving autonomy to your team while offering guidance when needed.",
    "You break down IT solutions in a way that even non-technical coworkers can understand.",
    "You are a master at fostering a culture of continuous improvement within your team.",
    "You have a knack for developing leadership potential in others through your mentorship.",
    "You lead with empathy, always considering the well-being of your team members.",
    "You are excellent at setting clear, achievable goals that drive performance.",
    "You always find a way to make complicated systems easier for others to navigate.",
    "You guide your team through change management with clarity and patience.",
    "You know how to turn abstract ideas into concrete project plans that everyone can follow.",
    "You excel at simplifying compliance and regulatory issues for your team.",
    "You consistently help others improve their presentation and communication skills.",
    "You are a champion of work-life balance, ensuring your team stays productive and healthy.",
    "You have a sharp eye for identifying gaps in workflows and implementing fixes.",
    "You create training materials that make learning new systems and tools enjoyable.",
    "You are known for your ability to build strong, high-performing teams from the ground up.",
    "You keep everyone aligned by ensuring transparency in communication and decision-making.",
    "You are skilled at developing strategies that maximize both efficiency and innovation.",
    "You are a trusted advisor, guiding coworkers through difficult decisions with clarity.",
    "You are great at nurturing a culture of feedback where everyone is encouraged to grow.",
    "You always have a contingency plan ready, ensuring your team is prepared for any scenario.",
    "You inspire others to adopt new technologies and tools to streamline their work.",
    "You help coworkers navigate cross-departmental collaborations with ease and diplomacy.",
    "You are always ahead of industry trends, guiding your team to stay competitive.",
    "You have a natural ability to foster a supportive and inclusive work environment.",
    "You consistently push your team to achieve ambitious goals with your motivational style.",
    "You are excellent at helping others see the big picture while managing day-to-day details.",
    "You excel at guiding remote teams to stay connected and productive, no matter the distance.",
    "You turn performance reviews into growth conversations that inspire improvement.",
    "You always find ways to optimize workflows, saving time and resources for your team.",
    "You are a strong advocate for employee development and continuous learning.",
    "You lead your team through crises with confidence, ensuring everyone stays focused."
]

motivational_phrases = [
    "This is important for my career.",
    "I really enjoy working on stuff like this.",
    "I want to help my friends with this.",
    "It helps me grow as a person.",
    "I want to challenge myself.",
    "I take pride in doing things well.",
    "It's a great learning opportunity.",
    "I want to prove to myself that I can do it.",
    "This is a stepping stone to bigger goals.",
    "I love the sense of accomplishment when I finish.",
    "It's something I’ve always wanted to do.",
    "I believe in the cause behind this.",
    "I want to show others that I am reliable.",
    "This will open up more opportunities for me.",
    "It’s a chance to develop new skills.",
    "I like to finish what I start.",
    "I want to make a difference.",
    "This will bring me closer to my long-term goals.",
    "It's part of my self-discipline.",
    "I enjoy pushing myself to the next level.",
    "I want to see what I’m capable of.",
    "This aligns with my values and beliefs.",
    "I enjoy the creative process involved.",
    "I’m passionate about this subject.",
    "It's a way to contribute to something bigger.",
    "I enjoy problem-solving.",
    "This will improve my expertise in the field.",
    "I want to set a good example for others.",
    "It’s a task that needs to be done, and I want to be responsible.",
    "I appreciate the recognition for a job well done.",
    "This is a great opportunity for personal growth.",
    "I like the satisfaction of completing something difficult.",
    "It brings me one step closer to my dreams.",
    "I want to prove others wrong.",
    "It’s a way for me to give back.",
    "I’m excited to see the end result.",
    "It’s a project that aligns with my personal interests.",
    "I like the process of getting things done.",
    "I want to exceed expectations.",
    "It's important for my personal development.",
    "I enjoy the challenge.",
    "This task will improve my confidence.",
    "I want to feel productive.",
    "It’s a way to achieve my personal goals.",
    "I believe this will make a positive impact.",
    "I’m driven by my curiosity to see what will happen.",
    "I want to keep improving.",
    "It’s something I can be proud of when completed.",
    "I enjoy the feeling of progress.",
    "This will help me be more efficient in the future.",
    "I’m doing it for my own satisfaction."
]

cot_prompts = [
    "Let's break down the problem step-by-step to find a solution.",
    "First, let’s understand the goal and any sub-goals needed to solve this.",
    "To approach this problem, let's identify the main components and tackle them one by one.",
    "Let’s start by considering any similar problems and how they were solved.",
    "Let's list what we know and what we need to find out.",
    "What initial steps can we take to simplify this problem?",
    "Let's analyze the problem carefully and list any possible approaches.",
    "Let’s outline the steps we need to take to reach a solution.",
    "To solve this, let’s divide the problem into manageable parts.",
    "Let’s clarify each part of the problem and solve each step-by-step.",
    "Let's identify any assumptions we’re making and verify them.",
    "What logical steps should we follow to tackle this problem?",
    "Let’s start by reviewing the information given and noting any useful patterns.",
    "Let’s hypothesize a solution path and test it with the data provided.",
    "Let’s examine the problem for any underlying principles or rules.",
    "To proceed, let’s determine any constraints and how they might affect our solution.",
    "Let’s break down the problem into smaller questions we can answer first.",
    "To solve this, let’s identify what information is essential and what can be ignored.",
    "Let’s evaluate potential methods and consider the pros and cons of each.",
    "Let’s take a closer look at each component of the problem and verify our approach.",
    "Let's begin by understanding the problem fully. We should clarify what is being asked, identify any constraints or requirements, and consider any underlying assumptions we may have. Breaking down the question first will help us avoid unnecessary steps later.",
    "To solve this, let’s start by identifying any patterns or similarities with known problems. We can look for parts of the problem that remind us of simpler examples or solutions we've seen before, and see if those methods apply here. Often, simplifying a problem is a great first step.",
    "Let's think about the main goal and the possible paths to achieve it. We should outline the overall objective first, and then brainstorm any sub-tasks or smaller questions that need answering along the way. Each smaller question will guide us closer to the final answer.",
    "Let’s work through this problem by analyzing each piece of information separately. We'll assess what each piece means in the context of the problem, whether it suggests any initial steps, and how it might connect to other pieces. Understanding the role of each part will make it easier to approach the solution.",
    "Let's list any assumptions we are making and check if they hold true. Assumptions can shape how we interpret a problem, so we’ll want to ensure that each one is valid and that we aren't overlooking potential exceptions or alternative interpretations.",
    "To clarify our approach, let’s map out the problem into a logical sequence of steps. We'll determine an initial step to get started, and then outline each subsequent step based on the results we expect from the previous ones. This approach will help us stay organized and avoid confusion.",
    "Let's consider any constraints and explore how they impact our solution options. Constraints might limit certain approaches or prioritize others, so we’ll keep them in mind as we outline possible methods. Sometimes constraints can even suggest an efficient approach directly.",
    "Let's try making an educated guess or hypothesis and use it to test our understanding of the problem. By running a hypothetical scenario, we can often identify areas we need to refine or correct before we dive into detailed calculations or code. Testing a hypothesis first can save us time and effort later.",
    "To make the problem more manageable, let’s divide it into smaller, independent parts we can solve individually. Once each part is addressed, we’ll combine the solutions to see if they form a complete answer. Tackling one small piece at a time will make the overall solution clearer.",
    "Let’s take a moment to review our solution process at each step to ensure accuracy and alignment with the goal. After each step, we’ll double-check if our results still fit the requirements and if we’re moving toward the solution. This habit of reflection can help us catch mistakes early and keep our focus.",
    "Alright, first, let’s just get a handle on what this question is asking. Sometimes reading through the problem again slowly can help us spot any details we missed. Let’s make sure we totally understand the goal here.",
    "Let’s try breaking this down into simpler parts. If we can tackle one small piece at a time, it might not seem as overwhelming. We can always put the pieces together later!",
    "Maybe we can start by writing out what we know so far. We can list any facts, numbers, or details given in the question and think about how each one might fit into the solution.",
    "How about we try solving a super basic version of this problem first? Like, if we just pretend it’s a really simple case, we might see a pattern or get a hint about what we need to do for the real question.",
    "Let’s make a quick plan. We’ll jot down a few steps or ideas that come to mind and then see if they seem like they’ll get us closer to the answer. It doesn’t have to be perfect yet—just an outline to get us started.",
    "Before we jump into solving it, let’s think about whether there are any rules or formulas we should use here. Sometimes just identifying a formula or strategy can be half the battle.",
    "We could try working backwards from the answer we want. Like, if we know where we need to end up, maybe we can figure out a path back to the start. Working backwards can sometimes make things way clearer!",
    "Okay, let’s think about if there are any things we’re assuming without realizing it. Sometimes, we assume a certain fact is true without questioning it, which could steer us in the wrong direction. Let's question everything and see what holds up.",
    "Let’s try to imagine what the solution would look like if we solved it. Sometimes, picturing the final answer can give us clues about how to start or what steps we need to take to get there.",
    "How about we go through this step-by-step together? We can try each step and then pause to check if it makes sense before moving on to the next one. Slow and steady can help us avoid mistakes!"
]

def random_length(points_range: tuple[int, int], sentences_per_point_range: tuple[int, int]) -> str:
        a, b = points_range
        points = random.randint(a,b)
        c, d = sentences_per_point_range 
        sentence_per_point = random.randint(c,d)
        sentence_instruction = f"up to {sentence_per_point} sentences" if sentence_per_point > 1 else "1 sentence"
        if points == 1:
            return f"Respond with {sentence_instruction}.\n"
        else:
            return f"Keep your response to {points} most important points with {sentence_instruction} for each point.\n"
