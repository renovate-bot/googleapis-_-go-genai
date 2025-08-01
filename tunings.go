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

// Code generated by the Google Gen AI SDK generator DO NOT EDIT.

package genai

import (
	"context"
	"fmt"
	"iter"
	"log"
	"net/http"
	"strings"
	"sync"
)

func getTuningJobParametersToMldev(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromName := getValueByPath(fromObject, []string{"name"})
	if fromName != nil {
		setValueByPath(toObject, []string{"_url", "name"}, fromName)
	}

	fromConfig := getValueByPath(fromObject, []string{"config"})
	if fromConfig != nil {
		setValueByPath(toObject, []string{"config"}, fromConfig)
	}

	return toObject, nil
}

func listTuningJobsConfigToMldev(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromPageSize := getValueByPath(fromObject, []string{"pageSize"})
	if fromPageSize != nil {
		setValueByPath(parentObject, []string{"_query", "pageSize"}, fromPageSize)
	}

	fromPageToken := getValueByPath(fromObject, []string{"pageToken"})
	if fromPageToken != nil {
		setValueByPath(parentObject, []string{"_query", "pageToken"}, fromPageToken)
	}

	fromFilter := getValueByPath(fromObject, []string{"filter"})
	if fromFilter != nil {
		setValueByPath(parentObject, []string{"_query", "filter"}, fromFilter)
	}

	return toObject, nil
}

func listTuningJobsParametersToMldev(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromConfig := getValueByPath(fromObject, []string{"config"})
	if fromConfig != nil {
		fromConfig, err = listTuningJobsConfigToMldev(fromConfig.(map[string]any), toObject)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"config"}, fromConfig)
	}

	return toObject, nil
}

func tuningExampleToMldev(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromTextInput := getValueByPath(fromObject, []string{"textInput"})
	if fromTextInput != nil {
		setValueByPath(toObject, []string{"textInput"}, fromTextInput)
	}

	fromOutput := getValueByPath(fromObject, []string{"output"})
	if fromOutput != nil {
		setValueByPath(toObject, []string{"output"}, fromOutput)
	}

	return toObject, nil
}

func tuningDatasetToMldev(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)
	if getValueByPath(fromObject, []string{"gcsUri"}) != nil {
		return nil, fmt.Errorf("gcsUri parameter is not supported in Gemini API")
	}

	if getValueByPath(fromObject, []string{"vertexDatasetResource"}) != nil {
		return nil, fmt.Errorf("vertexDatasetResource parameter is not supported in Gemini API")
	}

	fromExamples := getValueByPath(fromObject, []string{"examples"})
	if fromExamples != nil {
		fromExamples, err = applyConverterToSlice(fromExamples.([]any), tuningExampleToMldev)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"examples", "examples"}, fromExamples)
	}

	return toObject, nil
}

func createTuningJobConfigToMldev(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	if getValueByPath(fromObject, []string{"validationDataset"}) != nil {
		return nil, fmt.Errorf("validationDataset parameter is not supported in Gemini API")
	}

	fromTunedModelDisplayName := getValueByPath(fromObject, []string{"tunedModelDisplayName"})
	if fromTunedModelDisplayName != nil {
		setValueByPath(parentObject, []string{"displayName"}, fromTunedModelDisplayName)
	}

	if getValueByPath(fromObject, []string{"description"}) != nil {
		return nil, fmt.Errorf("description parameter is not supported in Gemini API")
	}

	fromEpochCount := getValueByPath(fromObject, []string{"epochCount"})
	if fromEpochCount != nil {
		setValueByPath(parentObject, []string{"tuningTask", "hyperparameters", "epochCount"}, fromEpochCount)
	}

	fromLearningRateMultiplier := getValueByPath(fromObject, []string{"learningRateMultiplier"})
	if fromLearningRateMultiplier != nil {
		setValueByPath(toObject, []string{"tuningTask", "hyperparameters", "learningRateMultiplier"}, fromLearningRateMultiplier)
	}

	if getValueByPath(fromObject, []string{"exportLastCheckpointOnly"}) != nil {
		return nil, fmt.Errorf("exportLastCheckpointOnly parameter is not supported in Gemini API")
	}

	if getValueByPath(fromObject, []string{"adapterSize"}) != nil {
		return nil, fmt.Errorf("adapterSize parameter is not supported in Gemini API")
	}

	fromBatchSize := getValueByPath(fromObject, []string{"batchSize"})
	if fromBatchSize != nil {
		setValueByPath(parentObject, []string{"tuningTask", "hyperparameters", "batchSize"}, fromBatchSize)
	}

	fromLearningRate := getValueByPath(fromObject, []string{"learningRate"})
	if fromLearningRate != nil {
		setValueByPath(parentObject, []string{"tuningTask", "hyperparameters", "learningRate"}, fromLearningRate)
	}

	return toObject, nil
}

func createTuningJobParametersToMldev(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromBaseModel := getValueByPath(fromObject, []string{"baseModel"})
	if fromBaseModel != nil {
		setValueByPath(toObject, []string{"baseModel"}, fromBaseModel)
	}

	fromTrainingDataset := getValueByPath(fromObject, []string{"trainingDataset"})
	if fromTrainingDataset != nil {
		fromTrainingDataset, err = tuningDatasetToMldev(fromTrainingDataset.(map[string]any), toObject)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"tuningTask", "trainingData"}, fromTrainingDataset)
	}

	fromConfig := getValueByPath(fromObject, []string{"config"})
	if fromConfig != nil {
		fromConfig, err = createTuningJobConfigToMldev(fromConfig.(map[string]any), toObject)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"config"}, fromConfig)
	}

	return toObject, nil
}

func getTuningJobParametersToVertex(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromName := getValueByPath(fromObject, []string{"name"})
	if fromName != nil {
		setValueByPath(toObject, []string{"_url", "name"}, fromName)
	}

	fromConfig := getValueByPath(fromObject, []string{"config"})
	if fromConfig != nil {
		setValueByPath(toObject, []string{"config"}, fromConfig)
	}

	return toObject, nil
}

func listTuningJobsConfigToVertex(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromPageSize := getValueByPath(fromObject, []string{"pageSize"})
	if fromPageSize != nil {
		setValueByPath(parentObject, []string{"_query", "pageSize"}, fromPageSize)
	}

	fromPageToken := getValueByPath(fromObject, []string{"pageToken"})
	if fromPageToken != nil {
		setValueByPath(parentObject, []string{"_query", "pageToken"}, fromPageToken)
	}

	fromFilter := getValueByPath(fromObject, []string{"filter"})
	if fromFilter != nil {
		setValueByPath(parentObject, []string{"_query", "filter"}, fromFilter)
	}

	return toObject, nil
}

func listTuningJobsParametersToVertex(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromConfig := getValueByPath(fromObject, []string{"config"})
	if fromConfig != nil {
		fromConfig, err = listTuningJobsConfigToVertex(fromConfig.(map[string]any), toObject)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"config"}, fromConfig)
	}

	return toObject, nil
}

func tuningDatasetToVertex(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromGcsUri := getValueByPath(fromObject, []string{"gcsUri"})
	if fromGcsUri != nil {
		setValueByPath(parentObject, []string{"supervisedTuningSpec", "trainingDatasetUri"}, fromGcsUri)
	}

	fromVertexDatasetResource := getValueByPath(fromObject, []string{"vertexDatasetResource"})
	if fromVertexDatasetResource != nil {
		setValueByPath(parentObject, []string{"supervisedTuningSpec", "trainingDatasetUri"}, fromVertexDatasetResource)
	}

	if getValueByPath(fromObject, []string{"examples"}) != nil {
		return nil, fmt.Errorf("examples parameter is not supported in Vertex AI")
	}

	return toObject, nil
}

func tuningValidationDatasetToVertex(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromGcsUri := getValueByPath(fromObject, []string{"gcsUri"})
	if fromGcsUri != nil {
		setValueByPath(toObject, []string{"validationDatasetUri"}, fromGcsUri)
	}

	fromVertexDatasetResource := getValueByPath(fromObject, []string{"vertexDatasetResource"})
	if fromVertexDatasetResource != nil {
		setValueByPath(parentObject, []string{"supervisedTuningSpec", "trainingDatasetUri"}, fromVertexDatasetResource)
	}

	return toObject, nil
}

func createTuningJobConfigToVertex(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromValidationDataset := getValueByPath(fromObject, []string{"validationDataset"})
	if fromValidationDataset != nil {
		fromValidationDataset, err = tuningValidationDatasetToVertex(fromValidationDataset.(map[string]any), toObject)
		if err != nil {
			return nil, err
		}

		setValueByPath(parentObject, []string{"supervisedTuningSpec"}, fromValidationDataset)
	}

	fromTunedModelDisplayName := getValueByPath(fromObject, []string{"tunedModelDisplayName"})
	if fromTunedModelDisplayName != nil {
		setValueByPath(parentObject, []string{"tunedModelDisplayName"}, fromTunedModelDisplayName)
	}

	fromDescription := getValueByPath(fromObject, []string{"description"})
	if fromDescription != nil {
		setValueByPath(parentObject, []string{"description"}, fromDescription)
	}

	fromEpochCount := getValueByPath(fromObject, []string{"epochCount"})
	if fromEpochCount != nil {
		setValueByPath(parentObject, []string{"supervisedTuningSpec", "hyperParameters", "epochCount"}, fromEpochCount)
	}

	fromLearningRateMultiplier := getValueByPath(fromObject, []string{"learningRateMultiplier"})
	if fromLearningRateMultiplier != nil {
		setValueByPath(parentObject, []string{"supervisedTuningSpec", "hyperParameters", "learningRateMultiplier"}, fromLearningRateMultiplier)
	}

	fromExportLastCheckpointOnly := getValueByPath(fromObject, []string{"exportLastCheckpointOnly"})
	if fromExportLastCheckpointOnly != nil {
		setValueByPath(parentObject, []string{"supervisedTuningSpec", "exportLastCheckpointOnly"}, fromExportLastCheckpointOnly)
	}

	fromAdapterSize := getValueByPath(fromObject, []string{"adapterSize"})
	if fromAdapterSize != nil {
		setValueByPath(parentObject, []string{"supervisedTuningSpec", "hyperParameters", "adapterSize"}, fromAdapterSize)
	}

	if getValueByPath(fromObject, []string{"batchSize"}) != nil {
		return nil, fmt.Errorf("batchSize parameter is not supported in Vertex AI")
	}

	if getValueByPath(fromObject, []string{"learningRate"}) != nil {
		return nil, fmt.Errorf("learningRate parameter is not supported in Vertex AI")
	}

	return toObject, nil
}

func createTuningJobParametersToVertex(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromBaseModel := getValueByPath(fromObject, []string{"baseModel"})
	if fromBaseModel != nil {
		setValueByPath(toObject, []string{"baseModel"}, fromBaseModel)
	}

	fromTrainingDataset := getValueByPath(fromObject, []string{"trainingDataset"})
	if fromTrainingDataset != nil {
		fromTrainingDataset, err = tuningDatasetToVertex(fromTrainingDataset.(map[string]any), toObject)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"supervisedTuningSpec", "trainingDatasetUri"}, fromTrainingDataset)
	}

	fromConfig := getValueByPath(fromObject, []string{"config"})
	if fromConfig != nil {
		fromConfig, err = createTuningJobConfigToVertex(fromConfig.(map[string]any), toObject)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"config"}, fromConfig)
	}

	return toObject, nil
}

func tunedModelFromMldev(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromModel := getValueByPath(fromObject, []string{"name"})
	if fromModel != nil {
		setValueByPath(toObject, []string{"model"}, fromModel)
	}

	fromEndpoint := getValueByPath(fromObject, []string{"name"})
	if fromEndpoint != nil {
		setValueByPath(toObject, []string{"endpoint"}, fromEndpoint)
	}

	return toObject, nil
}

func tuningJobFromMldev(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromSdkHttpResponse := getValueByPath(fromObject, []string{"sdkHttpResponse"})
	if fromSdkHttpResponse != nil {
		setValueByPath(toObject, []string{"sdkHttpResponse"}, fromSdkHttpResponse)
	}

	fromName := getValueByPath(fromObject, []string{"name"})
	if fromName != nil {
		setValueByPath(toObject, []string{"name"}, fromName)
	}

	fromState := getValueByPath(fromObject, []string{"state"})
	if fromState != nil {
		fromState, err = tTuningJobStatus(fromState)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"state"}, fromState)
	}

	fromCreateTime := getValueByPath(fromObject, []string{"createTime"})
	if fromCreateTime != nil {
		setValueByPath(toObject, []string{"createTime"}, fromCreateTime)
	}

	fromStartTime := getValueByPath(fromObject, []string{"tuningTask", "startTime"})
	if fromStartTime != nil {
		setValueByPath(toObject, []string{"startTime"}, fromStartTime)
	}

	fromEndTime := getValueByPath(fromObject, []string{"tuningTask", "completeTime"})
	if fromEndTime != nil {
		setValueByPath(toObject, []string{"endTime"}, fromEndTime)
	}

	fromUpdateTime := getValueByPath(fromObject, []string{"updateTime"})
	if fromUpdateTime != nil {
		setValueByPath(toObject, []string{"updateTime"}, fromUpdateTime)
	}

	fromDescription := getValueByPath(fromObject, []string{"description"})
	if fromDescription != nil {
		setValueByPath(toObject, []string{"description"}, fromDescription)
	}

	fromBaseModel := getValueByPath(fromObject, []string{"baseModel"})
	if fromBaseModel != nil {
		setValueByPath(toObject, []string{"baseModel"}, fromBaseModel)
	}

	fromTunedModel := getValueByPath(fromObject, []string{"_self"})
	if fromTunedModel != nil {
		fromTunedModel, err = tunedModelFromMldev(fromTunedModel.(map[string]any), toObject)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"tunedModel"}, fromTunedModel)
	}

	fromDistillationSpec := getValueByPath(fromObject, []string{"distillationSpec"})
	if fromDistillationSpec != nil {
		setValueByPath(toObject, []string{"distillationSpec"}, fromDistillationSpec)
	}

	fromExperiment := getValueByPath(fromObject, []string{"experiment"})
	if fromExperiment != nil {
		setValueByPath(toObject, []string{"experiment"}, fromExperiment)
	}

	fromLabels := getValueByPath(fromObject, []string{"labels"})
	if fromLabels != nil {
		setValueByPath(toObject, []string{"labels"}, fromLabels)
	}

	fromPipelineJob := getValueByPath(fromObject, []string{"pipelineJob"})
	if fromPipelineJob != nil {
		setValueByPath(toObject, []string{"pipelineJob"}, fromPipelineJob)
	}

	fromSatisfiesPzi := getValueByPath(fromObject, []string{"satisfiesPzi"})
	if fromSatisfiesPzi != nil {
		setValueByPath(toObject, []string{"satisfiesPzi"}, fromSatisfiesPzi)
	}

	fromSatisfiesPzs := getValueByPath(fromObject, []string{"satisfiesPzs"})
	if fromSatisfiesPzs != nil {
		setValueByPath(toObject, []string{"satisfiesPzs"}, fromSatisfiesPzs)
	}

	fromServiceAccount := getValueByPath(fromObject, []string{"serviceAccount"})
	if fromServiceAccount != nil {
		setValueByPath(toObject, []string{"serviceAccount"}, fromServiceAccount)
	}

	fromTunedModelDisplayName := getValueByPath(fromObject, []string{"tunedModelDisplayName"})
	if fromTunedModelDisplayName != nil {
		setValueByPath(toObject, []string{"tunedModelDisplayName"}, fromTunedModelDisplayName)
	}

	return toObject, nil
}

func listTuningJobsResponseFromMldev(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromSdkHttpResponse := getValueByPath(fromObject, []string{"sdkHttpResponse"})
	if fromSdkHttpResponse != nil {
		setValueByPath(toObject, []string{"sdkHttpResponse"}, fromSdkHttpResponse)
	}

	fromNextPageToken := getValueByPath(fromObject, []string{"nextPageToken"})
	if fromNextPageToken != nil {
		setValueByPath(toObject, []string{"nextPageToken"}, fromNextPageToken)
	}

	fromTuningJobs := getValueByPath(fromObject, []string{"tunedModels"})
	if fromTuningJobs != nil {
		fromTuningJobs, err = applyConverterToSlice(fromTuningJobs.([]any), tuningJobFromMldev)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"tuningJobs"}, fromTuningJobs)
	}

	return toObject, nil
}

func tuningOperationFromMldev(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromSdkHttpResponse := getValueByPath(fromObject, []string{"sdkHttpResponse"})
	if fromSdkHttpResponse != nil {
		setValueByPath(toObject, []string{"sdkHttpResponse"}, fromSdkHttpResponse)
	}

	fromName := getValueByPath(fromObject, []string{"name"})
	if fromName != nil {
		setValueByPath(toObject, []string{"name"}, fromName)
	}

	fromMetadata := getValueByPath(fromObject, []string{"metadata"})
	if fromMetadata != nil {
		setValueByPath(toObject, []string{"metadata"}, fromMetadata)
	}

	fromDone := getValueByPath(fromObject, []string{"done"})
	if fromDone != nil {
		setValueByPath(toObject, []string{"done"}, fromDone)
	}

	fromError := getValueByPath(fromObject, []string{"error"})
	if fromError != nil {
		setValueByPath(toObject, []string{"error"}, fromError)
	}

	return toObject, nil
}

func tunedModelCheckpointFromVertex(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromCheckpointId := getValueByPath(fromObject, []string{"checkpointId"})
	if fromCheckpointId != nil {
		setValueByPath(toObject, []string{"checkpointId"}, fromCheckpointId)
	}

	fromEpoch := getValueByPath(fromObject, []string{"epoch"})
	if fromEpoch != nil {
		setValueByPath(toObject, []string{"epoch"}, fromEpoch)
	}

	fromStep := getValueByPath(fromObject, []string{"step"})
	if fromStep != nil {
		setValueByPath(toObject, []string{"step"}, fromStep)
	}

	fromEndpoint := getValueByPath(fromObject, []string{"endpoint"})
	if fromEndpoint != nil {
		setValueByPath(toObject, []string{"endpoint"}, fromEndpoint)
	}

	return toObject, nil
}

func tunedModelFromVertex(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromModel := getValueByPath(fromObject, []string{"model"})
	if fromModel != nil {
		setValueByPath(toObject, []string{"model"}, fromModel)
	}

	fromEndpoint := getValueByPath(fromObject, []string{"endpoint"})
	if fromEndpoint != nil {
		setValueByPath(toObject, []string{"endpoint"}, fromEndpoint)
	}

	fromCheckpoints := getValueByPath(fromObject, []string{"checkpoints"})
	if fromCheckpoints != nil {
		fromCheckpoints, err = applyConverterToSlice(fromCheckpoints.([]any), tunedModelCheckpointFromVertex)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"checkpoints"}, fromCheckpoints)
	}

	return toObject, nil
}

func tuningJobFromVertex(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromSdkHttpResponse := getValueByPath(fromObject, []string{"sdkHttpResponse"})
	if fromSdkHttpResponse != nil {
		setValueByPath(toObject, []string{"sdkHttpResponse"}, fromSdkHttpResponse)
	}

	fromName := getValueByPath(fromObject, []string{"name"})
	if fromName != nil {
		setValueByPath(toObject, []string{"name"}, fromName)
	}

	fromState := getValueByPath(fromObject, []string{"state"})
	if fromState != nil {
		fromState, err = tTuningJobStatus(fromState)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"state"}, fromState)
	}

	fromCreateTime := getValueByPath(fromObject, []string{"createTime"})
	if fromCreateTime != nil {
		setValueByPath(toObject, []string{"createTime"}, fromCreateTime)
	}

	fromStartTime := getValueByPath(fromObject, []string{"startTime"})
	if fromStartTime != nil {
		setValueByPath(toObject, []string{"startTime"}, fromStartTime)
	}

	fromEndTime := getValueByPath(fromObject, []string{"endTime"})
	if fromEndTime != nil {
		setValueByPath(toObject, []string{"endTime"}, fromEndTime)
	}

	fromUpdateTime := getValueByPath(fromObject, []string{"updateTime"})
	if fromUpdateTime != nil {
		setValueByPath(toObject, []string{"updateTime"}, fromUpdateTime)
	}

	fromError := getValueByPath(fromObject, []string{"error"})
	if fromError != nil {
		setValueByPath(toObject, []string{"error"}, fromError)
	}

	fromDescription := getValueByPath(fromObject, []string{"description"})
	if fromDescription != nil {
		setValueByPath(toObject, []string{"description"}, fromDescription)
	}

	fromBaseModel := getValueByPath(fromObject, []string{"baseModel"})
	if fromBaseModel != nil {
		setValueByPath(toObject, []string{"baseModel"}, fromBaseModel)
	}

	fromTunedModel := getValueByPath(fromObject, []string{"tunedModel"})
	if fromTunedModel != nil {
		fromTunedModel, err = tunedModelFromVertex(fromTunedModel.(map[string]any), toObject)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"tunedModel"}, fromTunedModel)
	}

	fromSupervisedTuningSpec := getValueByPath(fromObject, []string{"supervisedTuningSpec"})
	if fromSupervisedTuningSpec != nil {
		setValueByPath(toObject, []string{"supervisedTuningSpec"}, fromSupervisedTuningSpec)
	}

	fromTuningDataStats := getValueByPath(fromObject, []string{"tuningDataStats"})
	if fromTuningDataStats != nil {
		setValueByPath(toObject, []string{"tuningDataStats"}, fromTuningDataStats)
	}

	fromEncryptionSpec := getValueByPath(fromObject, []string{"encryptionSpec"})
	if fromEncryptionSpec != nil {
		setValueByPath(toObject, []string{"encryptionSpec"}, fromEncryptionSpec)
	}

	fromPartnerModelTuningSpec := getValueByPath(fromObject, []string{"partnerModelTuningSpec"})
	if fromPartnerModelTuningSpec != nil {
		setValueByPath(toObject, []string{"partnerModelTuningSpec"}, fromPartnerModelTuningSpec)
	}

	fromDistillationSpec := getValueByPath(fromObject, []string{"distillationSpec"})
	if fromDistillationSpec != nil {
		setValueByPath(toObject, []string{"distillationSpec"}, fromDistillationSpec)
	}

	fromExperiment := getValueByPath(fromObject, []string{"experiment"})
	if fromExperiment != nil {
		setValueByPath(toObject, []string{"experiment"}, fromExperiment)
	}

	fromLabels := getValueByPath(fromObject, []string{"labels"})
	if fromLabels != nil {
		setValueByPath(toObject, []string{"labels"}, fromLabels)
	}

	fromPipelineJob := getValueByPath(fromObject, []string{"pipelineJob"})
	if fromPipelineJob != nil {
		setValueByPath(toObject, []string{"pipelineJob"}, fromPipelineJob)
	}

	fromSatisfiesPzi := getValueByPath(fromObject, []string{"satisfiesPzi"})
	if fromSatisfiesPzi != nil {
		setValueByPath(toObject, []string{"satisfiesPzi"}, fromSatisfiesPzi)
	}

	fromSatisfiesPzs := getValueByPath(fromObject, []string{"satisfiesPzs"})
	if fromSatisfiesPzs != nil {
		setValueByPath(toObject, []string{"satisfiesPzs"}, fromSatisfiesPzs)
	}

	fromServiceAccount := getValueByPath(fromObject, []string{"serviceAccount"})
	if fromServiceAccount != nil {
		setValueByPath(toObject, []string{"serviceAccount"}, fromServiceAccount)
	}

	fromTunedModelDisplayName := getValueByPath(fromObject, []string{"tunedModelDisplayName"})
	if fromTunedModelDisplayName != nil {
		setValueByPath(toObject, []string{"tunedModelDisplayName"}, fromTunedModelDisplayName)
	}

	return toObject, nil
}

func listTuningJobsResponseFromVertex(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
	toObject = make(map[string]any)

	fromSdkHttpResponse := getValueByPath(fromObject, []string{"sdkHttpResponse"})
	if fromSdkHttpResponse != nil {
		setValueByPath(toObject, []string{"sdkHttpResponse"}, fromSdkHttpResponse)
	}

	fromNextPageToken := getValueByPath(fromObject, []string{"nextPageToken"})
	if fromNextPageToken != nil {
		setValueByPath(toObject, []string{"nextPageToken"}, fromNextPageToken)
	}

	fromTuningJobs := getValueByPath(fromObject, []string{"tuningJobs"})
	if fromTuningJobs != nil {
		fromTuningJobs, err = applyConverterToSlice(fromTuningJobs.([]any), tuningJobFromVertex)
		if err != nil {
			return nil, err
		}

		setValueByPath(toObject, []string{"tuningJobs"}, fromTuningJobs)
	}

	return toObject, nil
}

type Tunings struct {
	apiClient *apiClient
}

func (m Tunings) get(ctx context.Context, name string, config *GetTuningJobConfig) (*TuningJob, error) {
	parameterMap := make(map[string]any)

	kwargs := map[string]any{"name": name, "config": config}
	deepMarshal(kwargs, &parameterMap)

	var httpOptions *HTTPOptions
	if config == nil || config.HTTPOptions == nil {
		httpOptions = &HTTPOptions{}
	} else {
		httpOptions = config.HTTPOptions
	}
	if httpOptions.Headers == nil {
		httpOptions.Headers = http.Header{}
	}
	var response = new(TuningJob)
	var responseMap map[string]any
	var fromConverter func(map[string]any, map[string]any) (map[string]any, error)
	var toConverter func(map[string]any, map[string]any) (map[string]any, error)
	if m.apiClient.clientConfig.Backend == BackendVertexAI {
		toConverter = getTuningJobParametersToVertex
		fromConverter = tuningJobFromVertex
	} else {
		toConverter = getTuningJobParametersToMldev
		fromConverter = tuningJobFromMldev
	}

	body, err := toConverter(parameterMap, nil)
	if err != nil {
		return nil, err
	}
	var path string
	var urlParams map[string]any
	if _, ok := body["_url"]; ok {
		urlParams = body["_url"].(map[string]any)
		delete(body, "_url")
	}
	if m.apiClient.clientConfig.Backend == BackendVertexAI {
		path, err = formatMap("{name}", urlParams)
	} else {
		path, err = formatMap("{name}", urlParams)
	}
	if err != nil {
		return nil, fmt.Errorf("invalid url params: %#v.\n%w", urlParams, err)
	}
	if _, ok := body["_query"]; ok {
		query, err := createURLQuery(body["_query"].(map[string]any))
		if err != nil {
			return nil, err
		}
		path += "?" + query
		delete(body, "_query")
	}

	if _, ok := body["config"]; ok {
		delete(body, "config")
	}
	responseMap, err = sendRequest(ctx, m.apiClient, path, http.MethodGet, body, httpOptions)
	if err != nil {
		return nil, err
	}
	responseMap, err = fromConverter(responseMap, nil)
	if err != nil {
		return nil, err
	}
	err = mapToStruct(responseMap, response)
	if err != nil {
		return nil, err
	}

	return response, nil
}

func (m Tunings) list(ctx context.Context, config *ListTuningJobsConfig) (*ListTuningJobsResponse, error) {
	parameterMap := make(map[string]any)

	kwargs := map[string]any{"config": config}
	deepMarshal(kwargs, &parameterMap)

	var httpOptions *HTTPOptions
	if config == nil || config.HTTPOptions == nil {
		httpOptions = &HTTPOptions{}
	} else {
		httpOptions = config.HTTPOptions
	}
	if httpOptions.Headers == nil {
		httpOptions.Headers = http.Header{}
	}
	var response = new(ListTuningJobsResponse)
	var responseMap map[string]any
	var fromConverter func(map[string]any, map[string]any) (map[string]any, error)
	var toConverter func(map[string]any, map[string]any) (map[string]any, error)
	if m.apiClient.clientConfig.Backend == BackendVertexAI {
		toConverter = listTuningJobsParametersToVertex
		fromConverter = listTuningJobsResponseFromVertex
	} else {
		toConverter = listTuningJobsParametersToMldev
		fromConverter = listTuningJobsResponseFromMldev
	}

	body, err := toConverter(parameterMap, nil)
	if err != nil {
		return nil, err
	}
	var path string
	var urlParams map[string]any
	if _, ok := body["_url"]; ok {
		urlParams = body["_url"].(map[string]any)
		delete(body, "_url")
	}
	if m.apiClient.clientConfig.Backend == BackendVertexAI {
		path, err = formatMap("tuningJobs", urlParams)
	} else {
		path, err = formatMap("tunedModels", urlParams)
	}
	if err != nil {
		return nil, fmt.Errorf("invalid url params: %#v.\n%w", urlParams, err)
	}
	if _, ok := body["_query"]; ok {
		query, err := createURLQuery(body["_query"].(map[string]any))
		if err != nil {
			return nil, err
		}
		path += "?" + query
		delete(body, "_query")
	}

	if _, ok := body["config"]; ok {
		delete(body, "config")
	}
	responseMap, err = sendRequest(ctx, m.apiClient, path, http.MethodGet, body, httpOptions)
	if err != nil {
		return nil, err
	}
	responseMap, err = fromConverter(responseMap, nil)
	if err != nil {
		return nil, err
	}
	err = mapToStruct(responseMap, response)
	if err != nil {
		return nil, err
	}

	return response, nil
}

func (m Tunings) tune(ctx context.Context, baseModel string, trainingDataset *TuningDataset, config *CreateTuningJobConfig) (*TuningJob, error) {
	parameterMap := make(map[string]any)

	kwargs := map[string]any{"baseModel": baseModel, "trainingDataset": trainingDataset, "config": config}
	deepMarshal(kwargs, &parameterMap)

	var httpOptions *HTTPOptions
	if config == nil || config.HTTPOptions == nil {
		httpOptions = &HTTPOptions{}
	} else {
		httpOptions = config.HTTPOptions
	}
	if httpOptions.Headers == nil {
		httpOptions.Headers = http.Header{}
	}
	var response = new(TuningJob)
	var responseMap map[string]any
	var fromConverter func(map[string]any, map[string]any) (map[string]any, error)
	var toConverter func(map[string]any, map[string]any) (map[string]any, error)
	if m.apiClient.clientConfig.Backend == BackendVertexAI {
		toConverter = createTuningJobParametersToVertex
		fromConverter = tuningJobFromVertex
	} else {

		return nil, fmt.Errorf("method Tune is only supported in the Vertex AI client. You can choose to use Vertex AI by setting ClientConfig.Backend to BackendVertexAI.")

	}

	body, err := toConverter(parameterMap, nil)
	if err != nil {
		return nil, err
	}
	var path string
	var urlParams map[string]any
	if _, ok := body["_url"]; ok {
		urlParams = body["_url"].(map[string]any)
		delete(body, "_url")
	}
	if m.apiClient.clientConfig.Backend == BackendVertexAI {
		path, err = formatMap("tuningJobs", urlParams)
	} else {
		path, err = formatMap("None", urlParams)
	}
	if err != nil {
		return nil, fmt.Errorf("invalid url params: %#v.\n%w", urlParams, err)
	}
	if _, ok := body["_query"]; ok {
		query, err := createURLQuery(body["_query"].(map[string]any))
		if err != nil {
			return nil, err
		}
		path += "?" + query
		delete(body, "_query")
	}

	if _, ok := body["config"]; ok {
		delete(body, "config")
	}
	responseMap, err = sendRequest(ctx, m.apiClient, path, http.MethodPost, body, httpOptions)
	if err != nil {
		return nil, err
	}
	responseMap, err = fromConverter(responseMap, nil)
	if err != nil {
		return nil, err
	}
	err = mapToStruct(responseMap, response)
	if err != nil {
		return nil, err
	}

	return response, nil
}

func (m Tunings) tuneMldev(ctx context.Context, baseModel string, trainingDataset *TuningDataset, config *CreateTuningJobConfig) (*TuningOperation, error) {
	parameterMap := make(map[string]any)

	kwargs := map[string]any{"baseModel": baseModel, "trainingDataset": trainingDataset, "config": config}
	deepMarshal(kwargs, &parameterMap)

	var httpOptions *HTTPOptions
	if config == nil || config.HTTPOptions == nil {
		httpOptions = &HTTPOptions{}
	} else {
		httpOptions = config.HTTPOptions
	}
	if httpOptions.Headers == nil {
		httpOptions.Headers = http.Header{}
	}
	var response = new(TuningOperation)
	var responseMap map[string]any
	var fromConverter func(map[string]any, map[string]any) (map[string]any, error)
	var toConverter func(map[string]any, map[string]any) (map[string]any, error)
	if m.apiClient.clientConfig.Backend == BackendVertexAI {

		return nil, fmt.Errorf("method TuneMldev is only supported in the Gemini Developer client. You can choose to use Gemini Developer client by setting ClientConfig.Backend to BackendGeminiAPI.")

	} else {
		toConverter = createTuningJobParametersToMldev
		fromConverter = tuningOperationFromMldev
	}

	body, err := toConverter(parameterMap, nil)
	if err != nil {
		return nil, err
	}
	var path string
	var urlParams map[string]any
	if _, ok := body["_url"]; ok {
		urlParams = body["_url"].(map[string]any)
		delete(body, "_url")
	}
	if m.apiClient.clientConfig.Backend == BackendVertexAI {
		path, err = formatMap("None", urlParams)
	} else {
		path, err = formatMap("tunedModels", urlParams)
	}
	if err != nil {
		return nil, fmt.Errorf("invalid url params: %#v.\n%w", urlParams, err)
	}
	if _, ok := body["_query"]; ok {
		query, err := createURLQuery(body["_query"].(map[string]any))
		if err != nil {
			return nil, err
		}
		path += "?" + query
		delete(body, "_query")
	}

	if _, ok := body["config"]; ok {
		delete(body, "config")
	}
	responseMap, err = sendRequest(ctx, m.apiClient, path, http.MethodPost, body, httpOptions)
	if err != nil {
		return nil, err
	}
	responseMap, err = fromConverter(responseMap, nil)
	if err != nil {
		return nil, err
	}
	err = mapToStruct(responseMap, response)
	if err != nil {
		return nil, err
	}

	return response, nil
}

var experimentalWarningTuningsCreateOperation sync.Once

// Tune creates a tuning job resource.
func (t Tunings) Tune(ctx context.Context, baseModel string, trainingDataset *TuningDataset, config *CreateTuningJobConfig) (*TuningJob, error) {
	experimentalWarningTuningsCreateOperation.Do(func() {
		log.Println("The SDK's tuning implementation is experimental, and may change in future versions.")
	})
	if t.apiClient.clientConfig.Backend == BackendVertexAI {
		return t.tune(ctx, baseModel, trainingDataset, config)
	} else {
		operation, err := t.tuneMldev(ctx, baseModel, trainingDataset, config)
		if err != nil {
			return nil, err
		}
		if operation == nil {
			return nil, fmt.Errorf("operation is nil")
		}
		if operation.Metadata != nil {
			if tunedModel, ok := operation.Metadata["tunedModel"].(string); ok {
				return &TuningJob{
					Name:  tunedModel,
					State: JobStateQueued,
				}, nil
			}
		}

		if operation.Name == "" {
			return nil, fmt.Errorf("operation name is required")
		}

		parts := strings.Split(operation.Name, "/operations/")
		tunedModelName := parts[0]
		return &TuningJob{
			Name:  tunedModelName,
			State: JobStateQueued,
		}, nil
	}
}

// Get retrieves a tuning job resource.
func (t Tunings) Get(ctx context.Context, name string, config *GetTuningJobConfig) (*TuningJob, error) {
	return t.get(ctx, name, config)
}

// List retrieves a paginated list of tuning job resources.
func (t Tunings) List(ctx context.Context, config *ListTuningJobsConfig) (Page[TuningJob], error) {
	listFunc := func(ctx context.Context, config map[string]any) ([]*TuningJob, string, *HTTPResponse, error) {
		var c ListTuningJobsConfig
		if err := mapToStruct(config, &c); err != nil {
			return nil, "", nil, err
		}
		resp, err := t.list(ctx, &c)
		if err != nil {
			return nil, "", nil, err
		}
		return resp.TuningJobs, resp.NextPageToken, resp.SDKHTTPResponse, nil
	}
	c := make(map[string]any)
	deepMarshal(config, &c)
	return newPage(ctx, "tuningJobs", c, listFunc)
}

// All retrieves all tuning job resources.
//
// This method handles pagination internally, making multiple API calls as needed
// to fetch all entries. It returns an iterator that yields each tuning job
// entry one by one. You do not need to manage pagination
// tokens or make multiple calls to retrieve all data.
func (t Tunings) All(ctx context.Context) iter.Seq2[*TuningJob, error] {
	listFunc := func(ctx context.Context, config map[string]any) ([]*TuningJob, string, *HTTPResponse, error) {
		var c ListTuningJobsConfig
		if err := mapToStruct(config, &c); err != nil {
			return nil, "", nil, err
		}
		resp, err := t.list(ctx, &c)
		if err != nil {
			return nil, "", nil, err
		}
		return resp.TuningJobs, resp.NextPageToken, resp.SDKHTTPResponse, nil
	}
	p, err := newPage(ctx, "tuningJobs", map[string]any{}, listFunc)
	if err != nil {
		return yieldErrorAndEndIterator[TuningJob](err)
	}
	return p.all(ctx)
}
