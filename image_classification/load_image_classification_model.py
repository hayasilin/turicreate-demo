import turicreate as tc

predict_data = tc.image_analysis.load_images('./TestImages/Cat', with_path=True)
predict_data.explore()

model = tc.load_model('./ImageClassify.model')

prediction = model.predict(tc.SFrame(data=predict_data))
print('prediction=')
print(prediction)

# Log result should be:
# prediction=
# ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat']