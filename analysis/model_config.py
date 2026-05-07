[
   {
       "model_id": "gemini-3.1-flash-lite",
       "temperature": 1,
       "top_p": 0.95,
       "top_k": 64,
       "max_tokens": 8192,
       "thinking_mode": "high", # "minimal", "low", "medium", "high"
       "provider": "hyprlab"
   },
   {
       "model_id": "gemini-3.1-flash-lite",
       "temperature": 1,
       "top_p": 0.95,
       "top_k": 64,
       "max_tokens": 8192,
       "thinking_mode": "minimal", # "minimal", "low", "medium", "high"
       "provider": "hyprlab"
   },
   {
       "model_id": "gemini-3.1-pro",
       "temperature": 1,
       "top_p": 0.95,
       "top_k": 64,
       "max_tokens": 8192,
       "thinking_mode": "high", # "minimal", "low", "medium", "high"
       "provider": "hyprlab"
   },
   {
       "model_id": "gemini-3.1-pro",
       "temperature": 1,
       "top_p": 0.95,
       "top_k": 64,
       "max_tokens": 8192,
       "thinking_mode": "low", # "low", "medium", "high" - minimal is not supported for this model
       "provider": "hyprlab"
   },
    {
         "model_id": "google/gemma-4-E4B-it",
         "temperature": 1,
         "top_p": 0.95,
         "top_k": 64,
         "max_tokens": 8192,
         "thinking_mode": False,
         "provider": "transformers"
    },
   {
       "model_id": "google/gemma-4-E4B-it",
       "temperature": 1,
       "top_p": 0.95,
       "top_k": 64,
       "max_tokens": 8192,
       "thinking_mode": True,
       "provider": "transformers"
   },
    {
         "model_id": "google/gemma-4-E2B-it",
         "temperature": 1,
         "top_p": 0.95,
         "top_k": 64,
         "max_tokens": 8192,
         "thinking_mode": False,
         "provider": "transformers"
    },
   {
       "model_id": "google/gemma-4-E2B-it",
       "temperature": 1,
       "top_p": 0.95,
       "top_k": 64,
       "max_tokens": 8192,
       "thinking_mode": True,
       "provider": "transformers"
   },
   {
       "model_id": "MOSS-Audio-4B-Thinking",
       "temperature": 1,
       "top_p": 1.0,
       "top_k": 50,
       "max_tokens": 8192,
       "thinking_mode": True,
       "provider": "moss"
   },
      {
       "model_id": "MOSS-Audio-4B-Instruct",
       "temperature": 1,
       "top_p": 1.0,
       "top_k": 50,
       "max_tokens": 8192,
       "thinking_mode": False,
       "provider": "moss"
   },
      {
       "model_id": "MOSS-Audio-8B-Thinking",
       "temperature": 1,
       "top_p": 1.0,
       "top_k": 50,
       "max_tokens": 8192,
       "thinking_mode": True,
       "provider": "moss"
   },
      {
       "model_id": "MOSS-Audio-8B-Instruct",
       "temperature": 1,
       "top_p": 1.0,
       "top_k": 50,
       "max_tokens": 8192,
       "thinking_mode": False,
       "provider": "moss"
   }
]