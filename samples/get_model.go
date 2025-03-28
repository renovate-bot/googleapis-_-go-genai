// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package main contains the sample code for the GenerateContent API.
package main

/*
# For VertexAI Backend
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT={YOUR_PROJECT_ID}
export GOOGLE_CLOUD_LOCATION={YOUR_LOCATION}

# For GeminiAPI Backend
export GOOGLE_GENAI_USE_VERTEXAI=false
export GOOGLE_API_KEY={YOUR_API_KEY}

go run samples/get_models.go
*/

import (
	"context"
	"flag"
	"fmt"
	"log"

	"google.golang.org/genai"
)

func getModels(ctx context.Context) {
	client, err := genai.NewClient(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	config := genai.GetModelConfig{}
	if client.ClientConfig().Backend == genai.BackendVertexAI {
		fmt.Println("Calling VertexAI Backend...")
	} else {
		fmt.Println("Calling GeminiAPI Backend...")
	}
	fmt.Println("Get model example.")
	model, err := client.Models.Get(ctx, "gemini-1.5-pro", &config)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%#v\n", model)
}

func main() {
	ctx := context.Background()
	flag.Parse()
	getModels(ctx)
}
