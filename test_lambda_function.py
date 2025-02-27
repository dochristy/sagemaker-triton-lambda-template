import json
import pytest
import boto3
from unittest.mock import patch, MagicMock
from lambda_function import (
    get_warp_details,
    preprocess,
    lambda_handler,
    postprocess
)

# Test warp details function
def test_get_warp_details():
    input_data = {
        'warp_id': '123456',
        'timestamp': '2023-01-01T12:00:00Z',
        'parameters': {'param1': 'value1'}
    }
    
    result = get_warp_details(input_data)
    
    assert result['warp_id'] == '123456'
    assert result['timestamp'] == '2023-01-01T12:00:00Z'
    assert result['parameters'] == {'param1': 'value1'}

# Test preprocessing function
def test_preprocess():
    input_data = {
        'data': {'field1': 'value1', 'field2': 'value2'}
    }
    
    warp_details = {
        'warp_id': '123456',
        'timestamp': '2023-01-01T12:00:00Z'
    }
    
    result = preprocess(input_data, warp_details)
    
    assert result['cleaned_data'] == {'field1': 'value1', 'field2': 'value2'}
    assert result['metadata']['warp_id'] == '123456'
    assert result['metadata']['timestamp'] == '2023-01-01T12:00:00Z'
    assert result['metadata']['preprocessing_completed'] is True

# Test postprocessing function
def test_postprocess():
    processed_data = {
        'sagemaker_response': {'prediction': [0.8, 0.2]},
        'metadata': {'warp_id': '123456', 'timestamp': '2023-01-01T12:00:00Z'},
        'endpoint_info': {'endpoint_name': 'test-endpoint', 'target_model': 'face88.tar.gz'}
    }
    
    result = postprocess(processed_data)
    
    assert result['results'] == {'prediction': [0.8, 0.2]}
    assert result['metadata']['warp_id'] == '123456'
    assert result['endpoint_info']['target_model'] == 'face88.tar.gz'
    assert result['summary']['status'] == 'success'
    assert result['summary']['postprocessing_completed'] is True

# Mock the SageMaker runtime for testing
@patch('lambda_function.sagemaker_runtime')
def test_lambda_handler_with_sagemaker(mock_sagemaker):
    # Mock SageMaker response
    mock_response = {
        'Body': MagicMock()
    }
    mock_response['Body'].read.return_value = json.dumps({'prediction': [0.8, 0.2]}).encode('utf-8')
    mock_sagemaker.invoke_endpoint.return_value = mock_response
    
    # Test event with endpoint name
    event = {
        'body': json.dumps({
            'warp_id': '123456',
            'timestamp': '2023-01-01T12:00:00Z',
            'endpoint_name': 'test-triton-endpoint',
            'data': {'image': 'base64_encoded_image_data'},
            'parameters': {'param1': 'value1'}
        })
    }
    
    context = {}
    
    # Call the lambda handler
    response = lambda_handler(event, context)
    
    # Verify the SageMaker client was called correctly
    mock_sagemaker.invoke_endpoint.assert_called_once_with(
        EndpointName='test-triton-endpoint',
        TargetModel='face88.tar.gz',
        ContentType='application/json',
        Body=json.dumps({'image': 'base64_encoded_image_data'})
    )
    
    # Verify response format
    assert response['statusCode'] == 200
    assert 'body' in response
    assert 'headers' in response
    assert response['headers']['Content-Type'] == 'application/json'
    
    body = json.loads(response['body'])
    assert 'results' in body
    assert 'metadata' in body
    assert 'endpoint_info' in body
    assert body['results'] == {'prediction': [0.8, 0.2]}
    assert body['endpoint_info']['target_model'] == 'face88.tar.gz'

# Test lambda handler with missing endpoint name
def test_lambda_handler_missing_endpoint():
    # Event without endpoint name
    event = {
        'body': json.dumps({
            'warp_id': '123456',
            'timestamp': '2023-01-01T12:00:00Z',
            'data': {'image': 'base64_encoded_image_data'},
            'parameters': {'param1': 'value1'}
        })
    }
    
    context = {}
    
    # Call the lambda handler
    response = lambda_handler(event, context)
    
    # Verify error response
    assert response['statusCode'] == 500
    body = json.loads(response['body'])
    assert 'error' in body
    assert 'SageMaker endpoint name is required' in body['error']

# Test lambda handler with error
def test_lambda_handler_error():
    # Invalid JSON in body to trigger exception
    event = {
        'body': '{invalid-json'
    }
    
    context = {}
    
    response = lambda_handler(event, context)
    
    assert response['statusCode'] == 500
    assert 'body' in response
    assert 'headers' in response
    assert response['headers']['Content-Type'] == 'application/json'
    
    body = json.loads(response['body'])
    assert 'error' in body