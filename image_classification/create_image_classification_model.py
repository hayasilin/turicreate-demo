import turicreate as tc

data = tc.image_analysis.load_images('./PetImages', with_path=True)

data['label'] = data['path'].apply(lambda path: 'dog' if '/Dog' in path else 'cat')

data.explore()

train_data, test_data = data.random_split(0.8)

model = tc.image_classifier.create(train_data, target='label')

metrics = model.evaluate(test_data)
print('accuracy')
print(metrics['accuracy'])

model.save('./ImageClassification.model')

model.export_coreml('./ImageClassification.mlmodel')

prediction = model.predict(tc.SFrame(data=test_data[0:1]))
print('prediction')
print(prediction)

# Log result should be:
# prediction=
# ['cat']