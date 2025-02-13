import boto3
import os
import json
import re
from dotenv import load_dotenv

# Load AWS credentials from .env file
load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Region Configuration
DYNAMODB_REGION = "ap-southeast-1"
BEDROCK_REGION = "us-east-1"

# Initialize AWS Clients
dynamodb_client = boto3.client(
    "dynamodb",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=DYNAMODB_REGION
)

bedrock_client = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=BEDROCK_REGION
)

def get_all_incidents(table_name, limit=5):
    """Fetch all incidents from DynamoDB."""
    try:
        response = dynamodb_client.scan(
            TableName=table_name,
            Limit=limit
        )
        return response.get("Items", [])
    except Exception as e:
        return {"error": f"Error fetching incidents: {str(e)}"}

def format_incidents_for_prompt(incidents):
    """Format incidents data for generating a prompt."""
    formatted_incidents = []
    for incident in incidents:
        formatted_incidents.append({
            "Incident ID": incident.get("id", {}).get("S", "N/A"),
            "Title": incident.get("title", {}).get("S", "N/A"),
            "Status": incident.get("status", {}).get("S", "N/A"),
            "Priority": incident.get("priority", {}).get("S", "N/A"),
            "Urgency": incident.get("urgency", {}).get("S", "N/A"),
            "Category": incident.get("category", {}).get("S", "N/A"),
            "Affected Service": incident.get("affectedService", {}).get("S", "N/A"),
            "Root Cause": incident.get("rootCauseAnalysis", {}).get("S", "N/A")
        })
    return json.dumps(formatted_incidents, indent=2)

def invoke_bedrock(prompt):
    """Invoke Bedrock to process the prompt."""
    try:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=json.dumps(payload)
        )
        response_body = json.loads(response["body"].read())
        return response_body.get("content", [{}])[0].get("text", "")
    except Exception as e:
        return {"error": f"Error invoking Bedrock: {str(e)}"}

def determine_incident_limit(user_question):
    """Determine the limit of incidents based on the user query."""
    match = re.search(r"last (\d+) incidents", user_question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 5  # Default to 5 if not specified

def generate_user_query_prompt(incidents, user_question):
    """Generate the final prompt to send to Bedrock."""
    return f"""You are an AI assistant helping with incident management. Below are the most recent incidents:

{format_incidents_for_prompt(incidents)}

User Question: {user_question}

Please provide a clear, professional response."""

def lambda_handler(event, context):
    """AWS Lambda function entry point."""
    try:
        body = json.loads(event["body"])
        user_question = body.get("user_question", "")

        if not user_question:
            return {"statusCode": 400, "body": json.dumps({"error": "Please provide a user question"})}

        # Determine incident limit
        limit = determine_incident_limit(user_question)
        
        # Get incidents
        table_name = "dev-incidents"
        incidents = get_all_incidents(table_name, limit)

        if isinstance(incidents, dict) and "error" in incidents:
            return {"statusCode": 500, "body": json.dumps(incidents)}

        if not incidents:
            return {"statusCode": 404, "body": json.dumps({"error": "No incidents found"})}

        # Generate prompt for Bedrock
        prompt = generate_user_query_prompt(incidents, user_question)

        # Get response from Bedrock
        response_text = invoke_bedrock(prompt)

        return {
            "statusCode": 200,
            "body": json.dumps({"response": response_text})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
