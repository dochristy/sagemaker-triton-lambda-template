# Data Science Lambda Function Template

A template for AWS Lambda functions following a standardized structure for data scientists, with SageMaker Triton endpoint integration.

## Structure

The template includes:

1. **lambda_function.py** - Main function with modular sections:
   - Warp Details - Extract and process metadata
   - Preprocessing - Clean and prepare input data
   - Lambda Handler - Main function entrypoint with SageMaker Triton endpoint invocation
   - Postprocessing - Format and finalize results

2. **Tests**:
   - **test_lambda_function.py** - Unit tests with pytest and mocked SageMaker responses
   - **test_lambda_function_moto.py** - Integration tests with moto for AWS services

## Getting Started

### Prerequisites

- Python 3.8+
- AWS CLI configured (for deployment)
- IAM role with permissions for Lambda and SageMaker

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running Tests

Run unit tests:
```
pytest test_lambda_function.py -v
```

Run AWS integration tests:
```
pytest test_lambda_function_moto.py -v
```

## Extending the Template

### SageMaker Triton Integration

The Lambda function is configured to invoke a SageMaker Triton endpoint with a specific model target:
- It invokes an endpoint with `TargetModel="face88.tar.gz"`
- You can modify this target model in the lambda_handler function
- Sends data in JSON format to the endpoint

### Adding AWS Services

The moto tests demonstrate how to test with:
- S3 for data storage/retrieval
- DynamoDB for result storage

Extend lambda_function.py to implement additional AWS service interactions.

### Customizing for Your Project

1. Modify the warp_details function to extract your specific metadata
2. Update preprocessing to handle your data format (images, text, etc.)
3. Adjust the SageMaker endpoint invocation in lambda_handler:
   - Change the TargetModel parameter to use your model file
   - Update ContentType as needed (application/json, application/octet-stream, etc.)
4. Customize postprocessing for your model's output format

## Deployment

1. Install required packages:
```
pip install boto3 -t .
```

2. Package the lambda function:
```
zip -r lambda_function.zip lambda_function.py ./boto3 ./botocore ./s3transfer ./jmespath ./python_dateutil ./six.py ./urllib3
```

3. Deploy using AWS CLI:
```
aws lambda create-function \
  --function-name your-sagemaker-triton-lambda \
  --runtime python3.9 \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --role your-lambda-execution-role-arn \
  --timeout 30 \
  --environment Variables={SAGEMAKER_ENDPOINT_NAME=your-triton-endpoint-name}
```

4. Make sure your Lambda execution role has the required permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": "*"
        }
    ]
}
```

## Best Practices

- Keep each section focused on its specific responsibility
- Write comprehensive tests for each section (mocking SageMaker responses)
- Use environment variables for configuration (endpoint names, model targets)
- Add appropriate logging throughout the code (especially for model responses)
- Implement proper error handling for SageMaker-specific errors
- Consider implementing retries for transient SageMaker endpoint failures
- Monitor performance metrics with CloudWatch
- Use AWS X-Ray for tracing request flows through your Lambda to SageMaker