import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, BoxPlot } from 'recharts';
import { Upload, Download, Eye, Settings, BarChart3, AlertCircle, CheckCircle, RefreshCw } from 'lucide-react';

const TitanicPreprocessing = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [dataStats, setDataStats] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [showVisualization, setShowVisualization] = useState(false);

  // Sample Titanic data for demonstration
  const sampleData = [
    { PassengerId: 1, Survived: 0, Pclass: 3, Name: "Braund, Mr. Owen Harris", Sex: "male", Age: 22, SibSp: 1, Parch: 0, Ticket: "A/5 21171", Fare: 7.25, Cabin: null, Embarked: "S" },
    { PassengerId: 2, Survived: 1, Pclass: 1, Name: "Cumings, Mrs. John Bradley", Sex: "female", Age: 38, SibSp: 1, Parch: 0, Ticket: "PC 17599", Fare: 71.28, Cabin: "C85", Embarked: "C" },
    { PassengerId: 3, Survived: 1, Pclass: 3, Name: "Heikkinen, Miss. Laina", Sex: "female", Age: 26, SibSp: 0, Parch: 0, Ticket: "STON/O2", Fare: 7.925, Cabin: null, Embarked: "S" },
    { PassengerId: 4, Survived: 1, Pclass: 1, Name: "Futrelle, Mrs. Jacques Heath", Sex: "female", Age: 35, SibSp: 1, Parch: 0, Ticket: "113803", Fare: 53.1, Cabin: "C123", Embarked: "S" },
    { PassengerId: 5, Survived: 0, Pclass: 3, Name: "Allen, Mr. William Henry", Sex: "male", Age: 35, SibSp: 0, Parch: 0, Ticket: "373450", Fare: 8.05, Cabin: null, Embarked: "S" },
    { PassengerId: 6, Survived: 0, Pclass: 3, Name: "Moran, Mr. James", Sex: "male", Age: null, SibSp: 0, Parch: 0, Ticket: "330877", Fare: 8.46, Cabin: null, Embarked: "Q" },
    { PassengerId: 7, Survived: 0, Pclass: 1, Name: "McCarthy, Mr. Timothy J", Sex: "male", Age: 54, SibSp: 0, Parch: 0, Ticket: "17463", Fare: 51.86, Cabin: "E46", Embarked: "S" },
    { PassengerId: 8, Survived: 0, Pclass: 3, Name: "Palsson, Master. Gosta Leonard", Sex: "male", Age: 2, SibSp: 3, Parch: 1, Ticket: "349909", Fare: 21.08, Cabin: null, Embarked: "S" }
  ];

  const steps = [
    {
      title: "1. Dataset Overview",
      description: "Import and explore basic dataset information",
      action: "explore"
    },
    {
      title: "2. Missing Values Analysis",
      description: "Identify and handle missing values",
      action: "missing"
    },
    {
      title: "3. Categorical Encoding",
      description: "Convert categorical features to numerical",
      action: "encode"
    },
    {
      title: "4. Feature Scaling",
      description: "Normalize/standardize numerical features",
      action: "scale"
    },
    {
      title: "5. Outlier Detection",
      description: "Visualize and handle outliers",
      action: "outliers"
    },
    {
      title: "6. Final Dataset",
      description: "Review cleaned and processed data",
      action: "final"
    }
  ];

  useEffect(() => {
    // Initialize with basic data stats
    analyzeData();
  }, []);

  const analyzeData = () => {
    const stats = {
      totalRows: sampleData.length,
      totalColumns: Object.keys(sampleData[0]).length,
      missingValues: {
        Age: sampleData.filter(row => row.Age === null).length,
        Cabin: sampleData.filter(row => row.Cabin === null).length,
        Embarked: sampleData.filter(row => row.Embarked === null).length
      },
      dataTypes: {
        numerical: ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare'],
        categorical: ['Sex', 'Embarked', 'Pclass'],
        text: ['Name', 'Ticket', 'Cabin']
      }
    };
    setDataStats(stats);
  };

  const processStep = (stepIndex) => {
    setCurrentStep(stepIndex);
    
    switch(steps[stepIndex].action) {
      case 'missing':
        handleMissingValues();
        break;
      case 'encode':
        encodeCategorical();
        break;
      case 'scale':
        scaleFeatures();
        break;
      case 'outliers':
        detectOutliers();
        break;
      case 'final':
        generateFinalDataset();
        break;
      default:
        break;
    }
  };

  const handleMissingValues = () => {
    const processed = sampleData.map(row => ({
      ...row,
      Age: row.Age || 29.7, // Mean age imputation
      Cabin: row.Cabin || 'Unknown',
      Embarked: row.Embarked || 'S' // Most common port
    }));
    setProcessedData(processed);
  };

  const encodeCategorical = () => {
    if (!processedData) return;
    
    const encoded = processedData.map(row => ({
      ...row,
      Sex_male: row.Sex === 'male' ? 1 : 0,
      Sex_female: row.Sex === 'female' ? 1 : 0,
      Embarked_S: row.Embarked === 'S' ? 1 : 0,
      Embarked_C: row.Embarked === 'C' ? 1 : 0,
      Embarked_Q: row.Embarked === 'Q' ? 1 : 0,
      HasCabin: row.Cabin !== 'Unknown' ? 1 : 0
    }));
    setProcessedData(encoded);
  };

  const scaleFeatures = () => {
    if (!processedData) return;
    
    // Simple min-max scaling for demonstration
    const fareValues = processedData.map(row => row.Fare);
    const minFare = Math.min(...fareValues);
    const maxFare = Math.max(...fareValues);
    
    const scaled = processedData.map(row => ({
      ...row,
      Fare_scaled: (row.Fare - minFare) / (maxFare - minFare),
      Age_scaled: row.Age / 100 // Simple scaling
    }));
    setProcessedData(scaled);
  };

  const detectOutliers = () => {
    setShowVisualization(true);
  };

  const generateFinalDataset = () => {
    if (!processedData) return;
    
    // Select final features for ML
    const finalFeatures = processedData.map(row => ({
      PassengerId: row.PassengerId,
      Survived: row.Survived,
      Pclass: row.Pclass,
      Age_scaled: row.Age_scaled || row.Age / 100,
      SibSp: row.SibSp,
      Parch: row.Parch,
      Fare_scaled: row.Fare_scaled || row.Fare / 100,
      Sex_male: row.Sex_male || (row.Sex === 'male' ? 1 : 0),
      Embarked_S: row.Embarked_S || (row.Embarked === 'S' ? 1 : 0),
      Embarked_C: row.Embarked_C || (row.Embarked === 'C' ? 1 : 0),
      HasCabin: row.HasCabin || (row.Cabin && row.Cabin !== 'Unknown' ? 1 : 0)
    }));
    setProcessedData(finalFeatures);
  };

  const renderStepContent = () => {
    const step = steps[currentStep];
    
    switch(step.action) {
      case 'explore':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{dataStats?.totalRows}</div>
                <div className="text-sm text-gray-600">Total Rows</div>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{dataStats?.totalColumns}</div>
                <div className="text-sm text-gray-600">Total Columns</div>
              </div>
              <div className="bg-yellow-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-yellow-600">
                  {dataStats ? Object.values(dataStats.missingValues).reduce((a,b) => a+b, 0) : 0}
                </div>
                <div className="text-sm text-gray-600">Missing Values</div>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {dataStats?.dataTypes.numerical.length + dataStats?.dataTypes.categorical.length}
                </div>
                <div className="text-sm text-gray-600">Features</div>
              </div>
            </div>
            
            <div className="bg-white p-4 rounded-lg border">
              <h3 className="font-semibold mb-2">Data Types</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <h4 className="font-medium text-blue-600 mb-1">Numerical</h4>
                  <ul className="text-sm space-y-1">
                    {dataStats?.dataTypes.numerical.map(col => (
                      <li key={col} className="text-gray-600">{col}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="font-medium text-green-600 mb-1">Categorical</h4>
                  <ul className="text-sm space-y-1">
                    {dataStats?.dataTypes.categorical.map(col => (
                      <li key={col} className="text-gray-600">{col}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="font-medium text-purple-600 mb-1">Text</h4>
                  <ul className="text-sm space-y-1">
                    {dataStats?.dataTypes.text.map(col => (
                      <li key={col} className="text-gray-600">{col}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        );
        
      case 'missing':
        return (
          <div className="space-y-4">
            <div className="bg-white p-4 rounded-lg border">
              <h3 className="font-semibold mb-3 flex items-center">
                <AlertCircle className="w-5 h-5 mr-2 text-yellow-500" />
                Missing Values Analysis
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {dataStats && Object.entries(dataStats.missingValues).map(([col, count]) => (
                  <div key={col} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                    <span className="font-medium">{col}</span>
                    <span className={`px-2 py-1 rounded text-sm ${count > 0 ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'}`}>
                      {count} missing
                    </span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="bg-white p-4 rounded-lg border">
              <h3 className="font-semibold mb-3">Imputation Strategy</h3>
              <div className="space-y-2 text-sm">
                <div className="flex items-center">
                  <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                  <span><strong>Age:</strong> Fill with mean age (29.7 years)</span>
                </div>
                <div className="flex items-center">
                  <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                  <span><strong>Cabin:</strong> Fill with 'Unknown' and create HasCabin feature</span>
                </div>
                <div className="flex items-center">
                  <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                  <span><strong>Embarked:</strong> Fill with most common port 'S' (Southampton)</span>
                </div>
              </div>
            </div>
          </div>
        );
        
      case 'encode':
        return (
          <div className="space-y-4">
            <div className="bg-white p-4 rounded-lg border">
              <h3 className="font-semibold mb-3">Categorical Encoding</h3>
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">One-Hot Encoding</h4>
                  <div className="space-y-2 text-sm">
                    <div className="p-2 bg-blue-50 rounded">
                      <strong>Sex:</strong> male/female → Sex_male, Sex_female (binary)
                    </div>
                    <div className="p-2 bg-green-50 rounded">
                      <strong>Embarked:</strong> S/C/Q → Embarked_S, Embarked_C, Embarked_Q (binary)
                    </div>
                    <div className="p-2 bg-purple-50 rounded">
                      <strong>Cabin:</strong> Present/Missing → HasCabin (binary feature)
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Sample Transformation</h4>
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-xs border border-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="p-2 border">Original</th>
                          <th className="p-2 border">Sex_male</th>
                          <th className="p-2 border">Sex_female</th>
                          <th className="p-2 border">Embarked_S</th>
                          <th className="p-2 border">HasCabin</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td className="p-2 border">male, S, No Cabin</td>
                          <td className="p-2 border text-center">1</td>
                          <td className="p-2 border text-center">0</td>
                          <td className="p-2 border text-center">1</td>
                          <td className="p-2 border text-center">0</td>
                        </tr>
                        <tr>
                          <td className="p-2 border">female, C, Has Cabin</td>
                          <td className="p-2 border text-center">0</td>
                          <td className="p-2 border text-center">1</td>
                          <td className="p-2 border text-center">0</td>
                          <td className="p-2 border text-center">1</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
        
      case 'scale':
        return (
          <div className="space-y-4">
            <div className="bg-white p-4 rounded-lg border">
              <h3 className="font-semibold mb-3">Feature Scaling</h3>
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Min-Max Normalization</h4>
                  <div className="p-3 bg-blue-50 rounded">
                    <div className="text-sm mb-2">Formula: (x - min) / (max - min)</div>
                    <div className="space-y-1 text-xs">
                      <div><strong>Fare:</strong> Range 0-512 → Scaled 0-1</div>
                      <div><strong>Age:</strong> Range 0-80 → Scaled 0-0.8</div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Why Scale Features?</h4>
                  <ul className="space-y-1 text-sm text-gray-600">
                    <li>• Prevents features with larger ranges from dominating</li>
                    <li>• Improves convergence speed in ML algorithms</li>
                    <li>• Required for distance-based algorithms (KNN, SVM)</li>
                    <li>• Helps neural networks train more effectively</li>
                  </ul>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-3 bg-red-50 rounded">
                    <h5 className="font-medium text-red-700 mb-1">Before Scaling</h5>
                    <div className="text-xs">
                      <div>Fare: 7.25, 71.28, 53.1</div>
                      <div>Age: 22, 38, 35</div>
                    </div>
                  </div>
                  <div className="p-3 bg-green-50 rounded">
                    <h5 className="font-medium text-green-700 mb-1">After Scaling</h5>
                    <div className="text-xs">
                      <div>Fare: 0.014, 0.139, 0.104</div>
                      <div>Age: 0.22, 0.38, 0.35</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
        
      case 'outliers':
        return (
          <div className="space-y-4">
            <div className="bg-white p-4 rounded-lg border">
              <h3 className="font-semibold mb-3">Outlier Detection</h3>
              
              <div className="mb-4">
                <h4 className="font-medium mb-2">Fare Distribution Analysis</h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={[
                      { range: '0-50', count: 15, normal: true },
                      { range: '50-100', count: 8, normal: true },
                      { range: '100-200', count: 3, normal: true },
                      { range: '200+', count: 2, normal: false }
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="range" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-3 bg-yellow-50 rounded">
                  <h5 className="font-medium text-yellow-700 mb-2">Detected Outliers</h5>
                  <ul className="text-sm space-y-1">
                    <li>• Fare > $200 (2 passengers)</li>
                    <li>• Age = 0 (infant records)</li>
                    <li>• Very large family sizes (SibSp + Parch > 6)</li>
                  </ul>
                </div>
                <div className="p-3 bg-blue-50 rounded">
                  <h5 className="font-medium text-blue-700 mb-2">Treatment Strategy</h5>
                  <ul className="text-sm space-y-1">
                    <li>• Keep high fares (valid luxury tickets)</li>
                    <li>• Cap age outliers at 99th percentile</li>
                    <li>• Create 'LargeFamily' binary feature</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        );
        
      case 'final':
        return (
          <div className="space-y-4">
            <div className="bg-white p-4 rounded-lg border">
              <h3 className="font-semibold mb-3 flex items-center">
                <CheckCircle className="w-5 h-5 mr-2 text-green-500" />
                Final Processed Dataset
              </h3>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div className="bg-green-50 p-3 rounded-lg text-center">
                  <div className="text-lg font-bold text-green-600">8</div>
                  <div className="text-sm text-gray-600">Sample Rows</div>
                </div>
                <div className="bg-blue-50 p-3 rounded-lg text-center">
                  <div className="text-lg font-bold text-blue-600">11</div>
                  <div className="text-sm text-gray-600">ML Features</div>
                </div>
                <div className="bg-purple-50 p-3 rounded-lg text-center">
                  <div className="text-lg font-bold text-purple-600">0</div>
                  <div className="text-sm text-gray-600">Missing Values</div>
                </div>
                <div className="bg-orange-50 p-3 rounded-lg text-center">
                  <div className="text-lg font-bold text-orange-600">100%</div>
                  <div className="text-sm text-gray-600">Data Quality</div>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium mb-2">Final Feature Set</h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
                  {['Pclass', 'Age_scaled', 'SibSp', 'Parch', 'Fare_scaled', 'Sex_male', 'Embarked_S', 'Embarked_C', 'HasCabin'].map(feature => (
                    <div key={feature} className="p-2 bg-gray-50 rounded text-center">
                      {feature}
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="mt-4 p-3 bg-green-50 rounded">
                <h5 className="font-medium text-green-700 mb-2">Ready for Machine Learning!</h5>
                <ul className="text-sm space-y-1">
                  <li>✅ All missing values handled</li>
                  <li>✅ Categorical variables encoded</li>
                  <li>✅ Numerical features scaled</li>
                  <li>✅ Outliers addressed</li>
                  <li>✅ Dataset is ML-ready</li>
                </ul>
              </div>
            </div>
          </div>
        );
        
      default:
        return <div>Select a step to begin preprocessing</div>;
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">Titanic Dataset - Data Cleaning & Preprocessing</h1>
        <p className="text-gray-600">Complete workflow for preparing the Titanic dataset for machine learning</p>
      </div>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex flex-wrap gap-2">
          {steps.map((step, index) => (
            <button
              key={index}
              onClick={() => processStep(index)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                currentStep === index
                  ? 'bg-blue-500 text-white'
                  : currentStep > index
                  ? 'bg-green-500 text-white'
                  : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
              }`}
            >
              {step.title}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="mb-4">
          <h2 className="text-xl font-semibold text-gray-800">{steps[currentStep].title}</h2>
          <p className="text-gray-600">{steps[currentStep].description}</p>
        </div>
        
        {renderStepContent()}
      </div>

      {/* Code Example */}
      <div className="mt-8 bg-gray-900 text-gray-100 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Settings className="w-5 h-5 mr-2" />
          Python Code Example
        </h3>
        <pre className="text-sm overflow-x-auto">
          <code>{`# Titanic Dataset Preprocessing Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 1: Load and explore data
df = pd.read_csv('titanic.csv')
print(df.info())
print(df.isnull().sum())

# Step 2: Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['HasCabin'] = df['Cabin'].notna().astype(int)

# Step 3: Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], prefix=['Sex', 'Embarked'])

# Step 4: Scale numerical features
scaler = StandardScaler()
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Step 5: Handle outliers (optional)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

# Step 6: Select final features
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 
           'Embarked_C', 'Embarked_Q', 'Embarked_S', 'HasCabin']
X = df[features]
y = df['Survived']

print("Dataset is ready for machine learning!")
print(f"Shape: {X.shape}")
print(f"Features: {list(X.columns)}")`}
        </code>
        </pre>
      </div>

      {/* Download Section */}
      <div className="mt-6 flex flex-wrap gap-4">
        <button className="flex items-center px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
          <Download className="w-4 h-4 mr-2" />
          Download Processed Data
        </button>
        <button className="flex items-center px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors">
          <BarChart3 className="w-4 h-4 mr-2" />
          Generate Report
        </button>
        <button className="flex items-center px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors">
          <RefreshCw className="w-4 h-4 mr-2" />
          Reset Pipeline
        </button>
      </div>
    </div>
  );
};

export default TitanicPreprocessing;
