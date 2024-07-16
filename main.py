# from table_updates.main import main
# main()
from dotenv import load_dotenv
load_dotenv()

from backend.query_handler import question_with_chat_state
from genai_toolbox.helper_functions.string_helpers import write_to_file, retrieve_file
import json
    
table_name = 'vector_table'
question = "Explain point 2 in greater detail"

chat_state = [
    {
        "role": "user", 
        "content": "What is the relevance of encryption in regulation?"
    },
    {
        "role": "assistant",
        "content": """
Encryption plays a crucial role in regulation, particularly in the context of balancing security, privacy, and control over information. Historically, encryption has been a contentious issue in regulatory frameworks, as evidenced by the export laws in the 1990s that restricted the export of encryption algorithms above a certain cryptographic strength. These regulations were aimed at preventing the misuse of strong encryption by adversaries but were often seen as overly restrictive and counterproductive by technologists and privacy advocates [1].

The relevance of encryption in regulation can be broken down into several key aspects:

1. **National Security and Law Enforcement**: Governments often regulate encryption to ensure that law enforcement and intelligence agencies can access communications when necessary. This is driven by the fear that strong encryption could be used by malicious actors to conceal their activities, making it difficult for authorities to intercept and decipher communications [2]. However, this creates a tension between the need for security and the right to privacy.

2. **Privacy and Civil Liberties**: Encryption is fundamental to protecting the privacy of individuals and organizations. It ensures that sensitive information, such as personal communications, financial transactions, and proprietary business data, remains confidential and secure from unauthorized access. The debate around encryption regulation often centers on finding a balance between enabling government access for security purposes and protecting individual privacy rights [3].

3. **Technological Innovation and Economic Impact**: Overly restrictive encryption regulations can stifle innovation and economic growth. For example, the limitations on exporting strong encryption in the 1990s were seen as hindering the development and global competitiveness of the tech industry. Today, encryption is a critical component of many technologies, from secure messaging apps to online banking, and regulations need to consider the potential impact on technological advancement and economic activity [4].

4. **Global Standards and Interoperability**: Encryption regulations can vary significantly between countries, leading to challenges in creating globally interoperable systems. Companies operating internationally must navigate a complex landscape of regulatory requirements, which can complicate the development and deployment of secure technologies. Harmonizing encryption standards and regulations across borders is essential for ensuring that security measures are effective and universally applicable [5].

5. **Public Perception and Trust**: The regulation of encryption also affects public trust in technology and government. If people believe that their communications and data are not secure, they may be less likely to use digital services, which can have broader implications for digital adoption and trust in institutions. Conversely, transparent and balanced encryption regulations can enhance public confidence in the security and privacy of digital services [6].

In summary, encryption is a pivotal element in the regulatory landscape, influencing national security, privacy, technological innovation, global interoperability, and public trust. Effective regulation must strike a delicate balance between enabling legitimate government access and protecting the privacy and security of individuals and organizations.
        """
    }
]

response = question_with_chat_state(
    question=question,
    chat_state=chat_state,
    table_name=table_name,
)

print(json.dumps(response['similar_chunks'], indent=4))
print(response['llm_response'])

