# Reasons for Common API

I see this message format alot:

```
    messages=[

        {

            "role": "system",

            "content": "You are a calculator assistant. Use the calculate function to perform mathematical operations and provide the results."

        },

        {

            "role": "user",

            "content": user_prompt,

        }

    ]
```
Let's make a wrapper for this since we see this repeated all the time.
