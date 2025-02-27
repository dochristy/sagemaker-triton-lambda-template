import json
import boto3
import pytest
from moto import mock_aws
from unittest.mock import patch, MagicMock
from lambda_function import lambda_handler

# Mock S3 setup
@mock_aws
@patch('lambda_function.sagemaker_runtime')
def test_s3_integration(mock_sagemaker):
    # Create mock S3 bucket and upload test data
    s3 = boto3.client('s3')
    bucket_name = 'test-data-bucket'
    s3.create_bucket(Bucket=bucket_name)
    
    # Upload test data to S3
    test_data = {'sample': 'data', 'for': 'testing'}
    s3.put_object(
        Bucket=bucket_name,
        Key='test-data.json',
        Body=json.dumps(test_data)
    )
    
    # Mock SageMaker response
    mock_response = {
        'Body': MagicMock()
    }
    mock_response['Body'].read.return_value = json.dumps({'prediction': [0.8, 0.2]}).encode('utf-8')
    mock_sagemaker.invoke_endpoint.return_value = mock_response
    
    # Create a test event that would use this S3 data
    event = {
        'body': json.dumps({
            'warp_id': 's3-test-123',
            'timestamp': '2023-01-01T12:00:00Z',
            'endpoint_name': 'test-triton-endpoint',
            'data_source': {
                'type': 's3',
                'bucket': bucket_name,
                'key': 'test-data.json'
            },
            'data': {'s3_data': True}
        })
    }
    
    # This test is just a placeholder - you would need to modify your lambda function
    # to actually read from S3 based on the data_source parameters
    response = lambda_handler(event, {})
    
    assert response['statusCode'] == 200

# Mock DynamoDB setup
@mock_aws
@patch('lambda_function.sagemaker_runtime')
def test_dynamodb_integration(mock_sagemaker):
    # Create mock DynamoDB table
    dynamodb = boto3.resource('dynamodb')
    table_name = 'test-results-table'
    
    # Create the table
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {'AttributeName': 'warp_id', 'KeyType': 'HASH'},
            {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
        ],
        AttributeDefinitions=[
            {'AttributeName': 'warp_id', 'AttributeType': 'S'},
            {'AttributeName': 'timestamp', 'AttributeType': 'S'}
        ],
        ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
    )
    
    # Mock SageMaker response
    mock_response = {
        'Body': MagicMock()
    }
    mock_response['Body'].read.return_value = json.dumps({'prediction': [0.8, 0.2]}).encode('utf-8')
    mock_sagemaker.invoke_endpoint.return_value = mock_response
    
    # Create a test event
    event = {
        'body': json.dumps({
            'warp_id': 'dynamo-test-123',
            'timestamp': '2023-01-01T12:00:00Z',
            'endpoint_name': 'test-triton-endpoint',
            'data': {'field1': 'value1', 'field2': 'value2'},
            'storage': {
                'type': 'dynamodb',
                'table': table_name
            }
        })
    }
    
    # This test is just a placeholder - you would need to modify your lambda function
    # to actually write to DynamoDB based on the storage parameters
    response = lambda_handler(event, {})
    
    assert response['statusCode'] == 200

# Test with both S3 and DynamoDB mocks
@mock_aws
@patch('lambda_function.sagemaker_runtime')
def test_full_aws_integration(mock_sagemaker):
    # Set up S3
    s3 = boto3.client('s3')
    bucket_name = 'test-data-bucket'
    s3.create_bucket(Bucket=bucket_name)
    
    test_data = {'sample': 'data', 'for': 'testing'}
    s3.put_object(
        Bucket=bucket_name,
        Key='test-data.json',
        Body=json.dumps(test_data)
    )
    
    # Set up DynamoDB
    dynamodb = boto3.resource('dynamodb')
    table_name = 'test-results-table'
    
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {'AttributeName': 'warp_id', 'KeyType': 'HASH'},
            {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
        ],
        AttributeDefinitions=[
            {'AttributeName': 'warp_id', 'AttributeType': 'S'},
            {'AttributeName': 'timestamp', 'AttributeType': 'S'}
        ],
        ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
    )
    
    # Mock SageMaker response
    mock_response = {
        'Body': MagicMock()
    }
    mock_response['Body'].read.return_value = json.dumps({'prediction': [0.8, 0.2]}).encode('utf-8')
    mock_sagemaker.invoke_endpoint.return_value = mock_response
    
    # Create a test event that would use both services
    event = {
        'body': json.dumps({
            'warp_id': 'full-test-123',
            'timestamp': '2023-01-01T12:00:00Z',
            'endpoint_name': 'test-triton-endpoint',
            'data': {'combined': 'data'},
            'data_source': {
                'type': 's3',
                'bucket': bucket_name,
                'key': 'test-data.json'
            },
            'storage': {
                'type': 'dynamodb',
                'table': table_name
            }
        })
    }
    
    # This test is just a placeholder - you would need to modify your lambda function
    # to interact with both services
    response = lambda_handler(event, {})
    
    assert response['statusCode'] == 200