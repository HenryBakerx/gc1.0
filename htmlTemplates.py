css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

# Replace with your actual GitHub raw image URLs below
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://raw.githubusercontent.com/HenryBakerx/Bot-12.2/926de73c60d8a270c385cbf6800a6d388ddcccd2/images/citiz%20logo.png" 
             style="max-height:78px; max-width:78px; border-radius:50%; object-fit:cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://raw.githubusercontent.com/HenryBakerx/Bot-12.2/8b042999a70ff0a8a66d59e6f6cbe6a5952c3d1b/images/bouwer.png" 
             style="max-height:78px; max-width:78px; border-radius:50%; object-fit:cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''




