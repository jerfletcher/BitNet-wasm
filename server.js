const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 8000;

// Serve static files from the current directory
app.use(express.static(__dirname, {
    index: ['index.html']
}));

// Directory listing middleware
app.use((req, res, next) => {
    const fullPath = path.join(__dirname, req.path);
    
    fs.stat(fullPath, (err, stats) => {
        if (err) {
            return next();
        }
        
        if (stats.isDirectory()) {
            const indexFile = path.join(fullPath, 'index.html');
            
            // Check if index.html exists
            fs.access(indexFile, fs.constants.F_OK, (err) => {
                if (!err) {
                    return next(); // Let static middleware handle index.html
                }
                
                // Generate directory listing
                fs.readdir(fullPath, (err, files) => {
                    if (err) {
                        return res.status(500).send('Server error');
                    }
                    
                    const list = files.map(file => {
                        const href = req.path.endsWith('/') ? req.path + file : req.path + '/' + file;
                        return `<li><a href="${href}">${file}</a></li>`;
                    }).join('');
                    
                    const html = `
                        <html>
                        <head><title>Directory: ${req.path}</title></head>
                        <body>
                            <h1>Directory: ${req.path}</h1>
                            <ul>${list}</ul>
                        </body>
                        </html>
                    `;
                    
                    res.send(html);
                });
            });
        } else {
            next();
        }
    });
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
