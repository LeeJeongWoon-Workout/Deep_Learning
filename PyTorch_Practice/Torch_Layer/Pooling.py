image.shape
pool = F.max_pool2d(image, 2,2)
pool.shape

pool_arr = pool.numpy()
pool_arr.shape

image_arr.shape

plt.figure(figsize=(10,15))
plt.subplot(121)
plt.title('input')
plt.imshow(np.squeeze(image_arr), 'gray')
plt.subplot(122)
plt.title('Output')
plt.imshow(np.squeeze(pool_arr), 'gray')
plt.show()

