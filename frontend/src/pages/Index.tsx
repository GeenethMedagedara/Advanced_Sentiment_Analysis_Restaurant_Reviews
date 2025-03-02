
import ReviewForm from "@/components/ReviewForm";
import { useState } from "react";
import axios from 'axios';

const Index = () => {
  const [sentimentval, setSentimentval] = useState<{aspect: string; sentiment: string } | null>(null);
  const [review, setReview] = useState<string>('');
  const [explanationshap, setExplanationshap] = useState("");
  const [explanationlime, setExplanationlime] = useState("");

  const handleReviewSubmit = async (review: string, aspect: string) => {
    setReview(review);
    try {
      const response = await axios.post('/sentiment/predict', { review, aspect });
      if (response.status == 200) {
        console.log(response.data);
        setSentimentval({aspect: response.data.aspect, sentiment: response.data.sentiment});
        const resimg = await axios.post('/explain/explain_shap', { review, aspect });
        if(resimg.status == 200) {
          console.log(resimg.data);
          setExplanationshap(`data:image/png;base64,${resimg.data.explanation}`)
          const resimglime = await axios.post('/explain/explain_lime', { review, aspect });
          if(resimglime.status == 200) {
            console.log(resimglime.data);
            setExplanationlime(`data:image/png;base64,${resimglime.data.explanation}`)
          }
        }

      }
      if(response.data.success) {
        console.log(response.data);
      } 
    } catch (error) {
      alert(error)
    }
    // This is where you'll integrate with your ML backend
    console.log('Submitted review:', { review, aspect });

  };

  return (
    <div className="min-h-screen p-6 bg-gradient-to-b from-background to-muted">
      <div className="container mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Left Column - Review Form */}
        <div className="space-y-8 animate-fadeIn">
          <div className="text-center md:text-left space-y-2">
            <h1 className="text-4xl font-bold tracking-tight">Review Analysis</h1>
            <p className="text-muted-foreground">
              Share your experience and let our AI analyze specific aspects of your review
            </p>
          </div>

          <div className="backdrop-blur-sm bg-card/50 p-6 rounded-lg shadow-lg border border-border/50">
            <ReviewForm onSubmit={handleReviewSubmit} />
          </div>
          {sentimentval && (
            <div className="p-4 bg-card/50 rounded-lg border border-border/50 backdrop-blur-sm animate-fadeIn">
              <p className="text-sm text-muted-foreground">
                Latest Analysis:
              </p>
              <p className="mt-2">
              Sentiment for the review analyzed for <span className="font-semibold capitalize">{sentimentval.aspect}
                </span> aspect is: <span className="font-semibold">{sentimentval.sentiment || 'Not analyzed'}</span>
              </p>
              <p className="mt-1 text-sm text-muted-foreground italic">
                "{review}"
              </p>
            </div>
          )}
        </div>

        {/* Right Column - Information Sections */}
        <div className="space-y-12 animate-fadeIn">
          {/* ABSA Sentiment Section */}
          <div className="space-y-4">
            <h2 className="text-3xl font-semibold text-center md:text-left">Spacy word similarity</h2>
            <div className="aspect-video rounded-lg overflow-hidden shadow-lg">
              {/* <img 
                src="https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d"
                alt="ABSA Sentiment Visualization"
                className="w-full h-full object-cover"
              /> */}
              {explanationshap && <img src={explanationshap} alt="Explanation Visualization" className="w-full h-full object-cover"/>}
            </div>
          </div>

          {/* How Model Works Section */}
          <div className="space-y-4">
            <h2 className="text-3xl font-semibold text-center md:text-left">LIME explanation</h2>
            <div className="aspect-video rounded-lg overflow-hidden shadow-lg">
              {/* <img 
                src="https://images.unsplash.com/photo-1487058792275-0ad4aaf24ca7"
                alt="Model Workflow Visualization"
                className="w-full h-full object-cover"
              /> */}
              {explanationlime && <img src={explanationlime} alt="Explanation Visualization" className="w-full h-full object-cover"/>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
