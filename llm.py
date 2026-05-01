import os
from google import genai
from groq import Groq

# load environment variables from .env file if exists
from dotenv import load_dotenv
load_dotenv()


def generate_llm_report(predicted_rul, p_failure, risk, action):
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    prompt = f"""
You are an industrial predictive maintenance assistant.

Machine status:
- Predicted RUL: {predicted_rul:.2f} cycles
- Failure Probability: {p_failure * 100:.1f}%
- Risk Level: {risk}
- Suggested Action: {action}

Write a professional maintenance report with:
1. Condition summary
2. Risk explanation (not just restating the risk level)
3. Maintenance recommendation
4. Urgency level

Do not change the numbers.
Do not invent sensor values.
Do not include any meta information or disclaimers. Only provide the report content.
"""

    # ---------------- GEMINI FIRST ----------------
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text

        except Exception as e:
            print("Gemini failed, switching to Groq...", e)

    # ---------------- GROQ FALLBACK ----------------
    if groq_key:
        try:
            client = Groq(api_key=groq_key)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  
                messages=[
                    {"role": "system", "content": "You are an industrial predictive maintenance assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            return response.choices[0].message.content

        except Exception as e:
            print("Groq also failed...", e)

    # ---------------- FINAL FALLBACK ----------------
    return f"""
### Maintenance Report

**Condition Summary:**  
The machine is classified as **{risk}**.

**Predicted RUL:** {predicted_rul:.2f} cycles  
**Failure Probability:** {p_failure * 100:.1f}%  

**Recommended Action:**  
{action}
"""

#test
print(generate_llm_report(150.5, 0.35, "High", "Schedule maintenance within the next 50 cycles. Monitor closely."))