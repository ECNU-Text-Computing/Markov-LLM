import pandas as pd
import argparse
from openai import OpenAI
from tqdm import tqdm

def integrate_gpt4(client, data, output_path, max_retries=3):
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        retry_count = 0
        while retry_count < max_retries:
            try:
                completion = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": row['input'][:2000]
                                                    + "Please predict the disruptive value (between -1 and 1) of focal paper based on the above information."
                                                      "Only a value is required, nothing else should be output."}
                    ]
                )
                generated_text = completion.choices[0].message.content.strip()
                data.at[index, 'gpt4'] = generated_text
                data.to_csv(output_path, index=False)
                print(generated_text)
                break
            except Exception as e:
                print(f"Error at index {index}, attempt {retry_count + 1}: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Max retries reached at index {index}. Moving to next.")
        if retry_count >= max_retries:
            continue
    return data

def main():
    parser = argparse.ArgumentParser(description="Integrate GPT-4 into dataset.")
    parser.add_argument('--data', type=str, help="Path to the data file", required=True)
    args = parser.parse_args()

    client = OpenAI(
        base_url="https://oneapi.xty.app/v1",
        api_key="sk-a4ONsYrfRZR3VOhBF81f1d89156f430a887b4900C4Cc74C2"
    )

    val_data_path = f'data/split/{args.data}_True_Falseval.csv'
    val_data = pd.read_csv(val_data_path)
    print("Integrating GPT-4 with validation data...")
    output_path = val_data_path.replace('.csv', '_gpt4.csv')
    val_data = integrate_gpt4(client, val_data, output_path)

if __name__ == '__main__':
    main()