#!/bin/bash
URL="https://manas-mitra-optimal-orcin.vercel.app/api/chat"
MESSAGES=(
  "I got one question wrong on the quiz so I'm basically going to fail the whole course and everyone will think I'm stupid."
  "My roommate borrowed my charger three days ago and hasn't given it back and I don't know why but it's really bothering me more than it should."
  "I don't even know what I'm feeling honestly, like I'm stressed about exams but also just kind of numb about everything and I keep telling myself I should be more motivated but I just can't."
  "Yaar mujhe lagta hai main kabhi kuch achha nahi kar sakta"
  "I'm so done with everything, I just want this semester to be over"
)
for i in "${!MESSAGES[@]}"; do
  echo "--- Message $((i+1)) ---"
  echo "Payload: ${MESSAGES[$i]}"
  time curl -s -X POST "$URL" -H "Content-Type: application/json" -d "{\"message\": \"${MESSAGES[$i]}\"}"
  echo ""
done
