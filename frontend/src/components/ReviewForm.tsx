
import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";

interface ReviewFormProps {
  onSubmit: (review: string, aspect: string) => Promise<void>;
}

const ReviewForm = ({ onSubmit }: ReviewFormProps) => {
  const [review, setReview] = useState('');
  const [aspect, setAspect] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const aspects = ['food', 'service', 'restaurant', 'delivery', 'price'];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!review.trim() || !aspect) {
      toast.error('Please fill in all fields');
      return;
    }

    setIsLoading(true);
    try {
      await onSubmit(review, aspect);
      toast.success('Review submitted successfully');
      setReview('');
      setAspect('');
    } catch (error) {
      toast.error('Failed to submit review');
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6 w-full max-w-lg mx-auto">
      <div className="space-y-2">
        <label htmlFor="review" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
          Your Review
        </label>
        <Textarea
          id="review"
          placeholder="Share your experience..."
          className="min-h-[150px] resize-none transition-all duration-200 focus:ring-2 focus:ring-offset-2"
          value={review}
          onChange={(e) => setReview(e.target.value)}
          disabled={isLoading}
        />
      </div>

      <div className="space-y-2">
        <label htmlFor="aspect" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
          Select Aspect
        </label>
        <Select value={aspect} onValueChange={setAspect}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Choose an aspect" />
          </SelectTrigger>
          <SelectContent>
            {aspects.map((a) => (
              <SelectItem key={a} value={a} className="capitalize">
                {a}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Button
        type="submit"
        className="w-full transition-all duration-200 animate-fadeIn"
        disabled={isLoading}
      >
        {isLoading ? 'Analyzing...' : 'Submit Review'}
      </Button>
    </form>
  );
};

export default ReviewForm;
