# LaclauGPT 

LaclauGPT is a political science multimodal data collection and analysis pipeline. It is called LaclauGPT as a tribute to [Ernesto Laclau](https://en.wikipedia.org/wiki/Ernesto_Laclau).

LaclauGPT is developed by [Tomi Toivio](mailto:tomi.toivio@helsinki.fi) for three [Helsinki Hub on Emotions, Populism and Polarisation](https://www.helsinki.fi/en/researchgroups/emotions-populism-and-polarisation) research projects funded by the European Union:
* [CO3](https://www.co3socialcontract.eu/) researches the social contract. 
* [ENDURE](https://www.endure-project.org/) researches the world after the pandemic. 
* [PLEDGE](https://www.pledgeproject.eu/) researches grievance politics.

The pipeline was used to collect and analyze multimodal social media data related to the 2024 European parliament elections. Data was collected from TikTok and Instagram. Data collection started in 1st of May 2024 and continued until the election day in 9th of June 2024. Collection was based on usernames of official election candidates as well as hashtags and search queries related to the elections. Election data was collected for Bulgaria, Croatia, Finland, France, Germany, Hungary, Portugal, Spain and Sweden. Collected and analyzed data cannot be released yet due to GDPR. This open source version uses dummy data. 

## Multimodal Data Analysis

These data analysis scripts are published for research documentation. You cannot use these without modification.

These are used with [Ollama](https://ollama.com/) running on [CSC Puhti](https://docs.csc.fi/computing/systems-puhti/) supercomputer.

The scripts are submitted as [batch jobs](https://docs.csc.fi/computing/running/creating-job-scripts-puhti/) in a sequence:

1. puhti_preprocess.py - This extracts video frames with OpenCV, processes the with EasyOCR and extracts a Whisper transcript of the audio.

2. puhti_frame.py - This uses Llama to create a multimodal analysis of 1-6 extracted frames.

3. puhti_summary.py - This creates a Llama summary analysis based on the metadata, Whisper transcript and Llama multimodal analysis results.

Code for the [TikTok Scraper](https://github.com/TomiToivio/LaclauGPT-TikTok-Scraper) used to collect EP2024 data is also available.
