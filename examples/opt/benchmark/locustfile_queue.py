from locust import HttpUser, task
from json import JSONDecodeError


class GenerationUser(HttpUser):
    @task
    def generate(self):
        prompt = 'Question: What is the longest river on the earth? Answer:'
        for i in range(4, 8):
            data = {'max_tokens': 2**i, 'prompt': prompt}
            with self.client.post('/queue_generation', json=data, catch_response=True) as response:
                try:
                    if False: # len(response.json()['text']) + 1 < data['max_tokens'] :
                        response.failure('Response wrong')
                except JSONDecodeError:
                    response.failure('Response could not be decoded as JSON')
                except KeyError:
                    response.failure('Response did not contain expected key "text"')