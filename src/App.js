import React, { useState, useMemo, useEffect } from "react";
import {
  Box,
  Heading,
  Text,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Button,
  Input,
  Stack,
  ChakraProvider,
  useColorModeValue,
  Image,
} from "@chakra-ui/react";
import * as tf from "@tensorflow/tfjs";


function padSequences(sequences, padding = 'pre', value = 0, maxlen) {
  const result = [];

  if (maxlen === undefined) {
    maxlen = sequences.reduce((maxlen, seq) => Math.max(maxlen, seq.length), 0);
  }

  for (const seq of sequences) {
    if (padding === 'pre') {
      const padLen = maxlen - seq.length;
      result.push(new Array(padLen).fill(value).concat(seq));
    } else if (padding === 'post') {
      const padLen = maxlen - seq.length;
      result.push(seq.concat(new Array(padLen).fill(value)));
    } else {
      throw new Error(`Invalid padding type: ${padding}`);
    }
  }

  return result;
}

function LandingPage() {
  const [fileData, setFileData] = useState([]);
  const [fileName, setFileName] = useState("");
  const [isClassifying, setIsClassifying] = useState(false);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    setFileName(file.name);
    const reader = new FileReader();
    reader.readAsText(file);
    reader.onload = () => {
      const csvData = reader.result;
      const parsedData = csvData
        .trim()
        .split("\n")
        .map((row) => ({ text: row.split(",")[0], classification: "" }));
      setFileData(parsedData);
    };
  };

  const handleClassifyClick = async () => {
    setIsClassifying(true);
    try {
      const weightsUrl =
        "https://raw.githack.com/petenathaphat/test-file/main/group1-shard1of1.bin";
      const weightsResponse = await fetch(weightsUrl);
      const weightsArrayBuffer = await weightsResponse.arrayBuffer();
      const weights = new Float32Array(weightsArrayBuffer);
  
      const modelUrl =
        "https://raw.githubusercontent.com/petenathaphat/test-file/main/sentiment_model.json";
      const model = await tf.loadLayersModel(modelUrl);
  
      const inputTensor = tf.tensor(fileData.map((row) => row.text));

  
      const outputTensor = model.predict(inputTensor);
      const predictions = outputTensor.arraySync();
  
      const classifiedData = fileData.map((row, index) => ({
        text: row.text,
        classification: predictions[index] >= 0.5 ? "positive" : "negative",
      }));
  
      setFileData(classifiedData);
    } catch (error) {
      console.error(error);
    } finally {
      setIsClassifying(false);
    }
  };
  

  const downloadFile = () => {
    const csvContent =
      "data:text/csv;charset=utf-8," +
      fileData.map((row) => `${row.text},${row.classification}`).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "file.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};


const tableBgColor = useColorModeValue("gray.50", "gray.900");

return (
<ChakraProvider>
<Box p={4}>
<Stack spacing={4} mb={4}>
<Heading as="h1" size="xl">
Sentiment Analysis
</Heading>
<Text>
Upload a CSV file containing texts to be classified as positive or
negative.
</Text>
<Input type="file" onChange={handleFileSelect} />
{fileName && <Text fontWeight="bold">{fileName}</Text>}
<Button onClick={handleClassifyClick} isLoading={isClassifying}>
Classify
</Button>
</Stack>
{fileData.length > 0 && (
<>
<Heading as="h2" size="lg" mb={2}>
Result
</Heading>
<Table variant="simple">
<Thead>
<Tr>
<Th>Text</Th>
<Th>Classification</Th>
</Tr>
</Thead>
<Tbody bg={tableBgColor}>
{fileData.map((row, index) => (
<Tr key={index}>
<Td>{row.text}</Td>
<Td>{row.classification}</Td>
</Tr>
))}
</Tbody>
</Table>
<Button mt={4} onClick={downloadFile}>
Download Result
</Button>
</>
)}
<Box mt={4}>
</Box>
</Box>
</ChakraProvider>
);
}

export default LandingPage;
