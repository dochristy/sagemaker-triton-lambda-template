import json
import os
import boto3

# SageMaker runtime client for invoking endpoints
sagemaker_runtime = boto3.client('sagemaker-runtime')

# ==================== WARP DETAILS ====================

def get_warp_details(input_data):
    """
    Extract and process warp details from input data.
    
    Args:
        input_data (dict): The input data
        
    Returns:
        dict: Processed warp details
    """
    # Example implementation
    warp_details = {
        'warp_id': input_data.get('warp_id'),
        'timestamp': input_data.get('timestamp'),
        'parameters': input_data.get('parameters', {})
    }
    
    return warp_details

# ==================== PREPROCESSING ====================

def preprocess(input_data, warp_details):
    """
    Preprocess the input data before main processing.
    
    Args:
        input_data (dict): The input data
        warp_details (dict): Warp details
        
    Returns:
        dict: Preprocessed data
    """
    # Example implementation
    preprocessed_data = {
        'cleaned_data': input_data.get('data', {}),
        'metadata': {
            'warp_id': warp_details['warp_id'],
            'timestamp': warp_details['timestamp'],
            'preprocessing_completed': True
        }
    }
    
    return preprocessed_data

# ==================== LAMBDA HANDLER ====================

def lambda_handler(event, context):
    """
    Main Lambda function handler that invokes a SageMaker Triton endpoint.
    
    Args:
        event (dict): Input event data
        context (object): Lambda execution context
        
    Returns:
        dict: Response with processed data and status code
    """
    try:
        # Extract input data
        input_data = json.loads(event.get('body', '{}'))
        
        # Get warp details
        warp_details = get_warp_details(input_data)
        
        # Preprocess data
        preprocessed_data = preprocess(input_data, warp_details)
        
        # Get the SageMaker endpoint name from the event or use a default
        endpoint_name = input_data.get('endpoint_name', os.environ.get('SAGEMAKER_ENDPOINT_NAME'))
        if not endpoint_name:
            raise ValueError("SageMaker endpoint name is required")
        
        # Get the payload for the model
        payload = preprocessed_data.get('cleaned_data')
        
        # Invoke the SageMaker Triton endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            TargetModel="face88.tar.gz",  # Specific model file
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        
        # Process the response
        response_body = response['Body'].read().decode('utf-8')
        
        try:
            # Try to parse as JSON
            model_output = json.loads(response_body)
        except json.JSONDecodeError:
            # If not JSON, return as string
            model_output = response_body
        
        # Add the model output to the processed data
        processed_data = {
            'sagemaker_response': model_output,
            'metadata': preprocessed_data.get('metadata', {}),
            'endpoint_info': {
                'endpoint_name': endpoint_name,
                'target_model': 'face88.tar.gz'
            }
        }
        
        # Postprocess results
        results = postprocess(processed_data)
        
        # Return successful response
        return {
            'statusCode': 200,
            'body': json.dumps(results),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    except Exception as e:
        # Log error
        print(f"Error: {str(e)}")
        
        # Return error response
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        }

# ==================== POSTPROCESSING ====================

def postprocess(processed_data):
    """
    Postprocess the results after main processing.
    
    Args:
        processed_data (dict): The processed data including SageMaker response
        
    Returns:
        dict: Postprocessed results
    """
    # Process the SageMaker response further if needed
    sagemaker_response = processed_data.get('sagemaker_response', {})
    
    # Example implementation
    results = {
        'results': sagemaker_response,
        'metadata': processed_data.get('metadata', {}),
        'endpoint_info': processed_data.get('endpoint_info', {}),
        'summary': {
            'status': 'success',
            'postprocessing_completed': True
        }
    }
    
    return results