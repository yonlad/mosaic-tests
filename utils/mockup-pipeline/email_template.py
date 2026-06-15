"""HTML email template for Mirror of Eternity mockup emails."""

SUBJECT = "Your Mirror of Eternity Mosaic Portrait"

def get_email_html(name: str) -> str:
    """Return the HTML email body with the given name interpolated."""
    return f"""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0; padding:0; background-color:#f5f5f5; font-family:Georgia, 'Times New Roman', serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f5f5f5;">
<tr><td align="center" style="padding:30px 15px;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff; border-radius:4px; overflow:hidden;">

<!-- Header -->
<tr><td style="padding:40px 40px 20px 40px;">
<h1 style="margin:0; font-size:22px; color:#1a1a1a; font-weight:normal; letter-spacing:0.5px;">
Mirror of Eternity
</h1>
<p style="margin:8px 0 0 0; font-size:13px; color:#888; letter-spacing:1px; text-transform:uppercase;">
Michelangelo Pistoletto
</p>
</td></tr>

<!-- Body -->
<tr><td style="padding:20px 40px 30px 40px; font-size:16px; line-height:1.7; color:#333;">

<p>Dear {name},</p>

<p>Thank you for participating in Michelangelo Pistoletto's
<a href="https://thebass.org/art/michelangelo-pistoletto-mirror-of-eternity-2025/"
   style="color:#8B6914; text-decoration:underline;">Mirror of Eternity</a>
installation at The Bass Museum of Art.</p>

<p>Your unique mosaic portrait has been created from your photograph, composed
of thousands of tiny tiles that reflect the themes of connection and reflection
central to Pistoletto's work.</p>

<p>We are pleased to offer you the opportunity to acquire a high-quality print
of your mosaic portrait. Each print is produced on archival paper with
museum-grade pigment inks.</p>

<p>If you are interested in acquiring your print, or have any questions,
please contact
<a href="mailto:jacqueline@dminti.com" style="color:#8B6914; text-decoration:underline;">
jacqueline@dminti.com</a>.</p>

<blockquote style="margin:25px 0; padding:15px 20px; border-left:3px solid #8B6914; background:#faf8f3; font-style:italic; color:#555;">
"The mirror is the door through which the self encounters the other."<br>
<span style="font-style:normal; font-size:14px; color:#888;">&mdash; Michelangelo Pistoletto</span>
</blockquote>

</td></tr>

<!-- Signature -->
<tr><td style="padding:0 40px 20px 40px; font-size:14px; color:#666; line-height:1.6;">
<p style="margin:0;">Warm regards,<br>
The Mirror of Eternity Team</p>
</td></tr>

<!-- Artwork Details -->
<tr><td style="padding:10px 40px 20px 40px; font-size:12px; color:#999; border-top:1px solid #eee; line-height:1.6;">
<p style="margin:8px 0 0 0;">
<strong>Artwork:</strong> Mirror of Eternity, 2025<br>
<strong>Artist:</strong> Michelangelo Pistoletto<br>
<strong>Venue:</strong> The Bass Museum of Art, Miami Beach
</p>
</td></tr>

<!-- Mockup Image -->
<tr><td style="padding:20px 40px 40px 40px; text-align:center;">
<p style="font-size:13px; color:#888; margin:0 0 15px 0; text-transform:uppercase; letter-spacing:1px;">
Your Mosaic Portrait
</p>
<img src="cid:mosaic-portrait"
     alt="Your mosaic portrait"
     style="max-width:100%; height:auto; border-radius:4px; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
</td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""
