const uploadFile = (req, res) => {
    if (!req.file) {
      return res.status(400).send('No file uploaded.');
    }
    res.send(`File uploaded successfully! <a href="/uploads/${req.file.filename}">View File</a>`);
  };
  
  module.exports = { uploadFile };
  