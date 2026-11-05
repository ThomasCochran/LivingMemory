#Copyright (C) Thomas Cochran
from datetime import datetime
import sqlite3
import ollama# type: ignore
import re
import json
import math
import logging

logger = logging.getLogger(__name__)

# when no tokenizerModel is give, chars are 3 tokens each, 
# Chinese, Japanese and Korean symbols are 1 token each
CHARS_PER_TOKEN = 3
_CJK = re.compile(r"[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-䶿一-鿿가-힯豈-﫿]")



# < - - - - - [ LivingMemory ] - - - - - >

class LivingMemory:

    def __init__(self, dbPath:str = "livingMemory.db", model:str = None, tokenizerModel:str = None, contextWindow:int = 2048, timeout:int = 180):

        self.model = model
        self.tokenizerModel = tokenizerModel
        if tokenizerModel is not None:
            from transformers import AutoTokenizer# type: ignore
            self.tokenizerModel = AutoTokenizer.from_pretrained(tokenizerModel)

        # handle missing model
        if model is None:
            logger.warning("LivingMemory is best used with an AI model. Example: LivingMemory(model='qwen3:14b')")

        # contextWindow is split in half, input/output
        self.contextWindow = contextWindow
        if contextWindow < 1024:
            logger.warning("context window is less than 1024 tokens, this may be too small to for usful memory summaries")

        self.outputSize = contextWindow // 2
        self.inputSize = contextWindow - self.outputSize
        self._client = ollama.Client(timeout=timeout)
        self.dbPath = dbPath
        self.VALID_ROLES = {"user", "assistant", "system", "tool"}








# < - - - - - [ Ollama chat management ] - - - - - >

    # extra is an optinal way to hold Ollama messafe fields, like
    # {"tool_name": "remember"} or {"images": [...]}
    def addConv(self, table:str, role:str = "user", memory:str = "blank", extra:dict = None):
        if role not in self.VALID_ROLES:
            logger.error(f"Invalid role '{role}'. Must be one of {self.VALID_ROLES}")
            raise ValueError(f"Invalid role '{role}'. Must be one of {self.VALID_ROLES}")

        # extra goes in first, so it can't overwrite role and stuff
        msg = dict(extra) if extra else {}
        msg["role"] = role
        msg["content"] = memory

        # written by claude, fixes JSON TypeError problems, don't delete
        def _toJson(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump(mode="json", exclude_none=True)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        # write the memory to the conv tag
        self.writeMem(table, "conv", "Memories about the conversation", json.dumps(msg, default=_toJson))



    def rConv(self, table:str, latest:int = 0) -> list:
        return [json.loads(row[3]) for row in self.readMem(table, "conv", latest)]



    def flushConv(self, table:str, tags:list, keep:int = 0):
        rows = self.readMem(table, "conv")
        toFile = rows[:-keep] if keep > 0 else rows

        # Shouldn't flush nothing
        filled = [row for row in toFile if row[3].strip()]
        if not filled:
            return

        # turn JSON into a readable transcript for the filter model
        lines = []
        for row in filled:
            msg = json.loads(row[3])
            role = msg.get("role", "")
            if role == "tool":
                lines.append(f"[{msg.get('tool_name', 'tool')} result]: {msg.get('content', '')}")
                continue
            if msg.get("content"):
                lines.append(f"{role}: {msg['content']}")
            for call in msg.get("tool_calls") or []:
                fn = call.get("function", {})
                lines.append(f"{role} used {fn.get('name')}: {json.dumps(fn.get('arguments', {}), ensure_ascii=False)}")
        convMem = "\n\n".join(lines)

        # remember where the table ends, so a failure partway can undo the new memories
        conn = sqlite3.connect(self.dbPath)
        try:
            startKey = conn.execute(f"SELECT MAX(key) FROM {table}").fetchone()[0]
        finally:
            conn.close()

        # get model to tag everything
        # enumerate() for logging loading bar.
        try:
            for i, tag in enumerate(tags, 1):
                logger.info(f"flushConv [{table}]: {i}/{len(tags)} — '{tag}'")
                self.tagMem(table, tag, convMem)
        except Exception:

            # undo the tags already written, so a retry doesn't duplicate them
            conn = sqlite3.connect(self.dbPath)
            try:
                conn.execute(f"DELETE FROM {table} WHERE key > ? AND tag != 'conv'", (startKey,))
                conn.commit()
            finally:
                conn.close()
            raise

        # delete the old rows
        conn = sqlite3.connect(self.dbPath)
        try:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table} WHERE tag = 'conv' AND key <= ?", (toFile[-1][0],))
            conn.commit()
        finally:
            conn.close()






# < - - - - - [ Tag / Desc ] - - - - - >

    def cleanAll(self, table:str, drop:list = None, model:str = None, maxTok:int = 0, chopTime:int = 9, prompt:str = None):
        tags = self.getTags(table, drop)
        loop = 0
        maxLoop = len(tags)

        # run cleanTag() on every tag minus the tags in "drop"
        for tag in tags:
            logger.info(f"cleanAll [{table}]: {loop}/{maxLoop} — '{tag}'")
            self.cleanTag(table, tag, model, maxTok, chopTime, prompt)
            loop+=1
        logger.info(f"cleanAll [{table}]: done, {maxLoop} tag(s) cleaned")



    def cleanTag(self, table:str, tag:str, model:str = None, maxTok:int = 0, chopTime:int = 9, prompt:str = None):

        # get the description
        desc = self.getDesc(table, tag)

        # find all memories under the tag
        memories = self.readMem(table, tag)

        # skip everything if no memories
        if not memories:
            return

        # just re-tag the memories, if not enough info provided
        if model is None or maxTok == 0 or self.model is None:
            logger.warning("cleanTag() works best when arguments, 'model:str' and 'maxTok:int' are given")

            # delete original memories and run initial compression
            self.deleteTag(table, tag, 0)
            try:
                self.tagMem(table, tag, "\n\n".join(row[3] for row in memories), desc)
            except Exception:

                # re-upload old memories on error
                for row in memories:
                    self.writeMem(table, tag, desc, row[3], row[4])
                raise
            return

        # allow user to pass their own prompt into the model
        def _buildPrompt(text, memTok, maxTok, percentage):
            if prompt is None:
                return f""" you are a summary professional
This text is too long, it uses {memTok} tokens, but it needs to be less then {maxTok} tokens
re-write it using about {percentage} percent the words/tokens.
keeping what is most useful and on the topic of "{desc}" and throwing away the rest.
Respond with only the filtered memories from a 1st person point of view (1st person from the assistant\'s POV).

here is the text : {text}

Your summary here :
"""
            else:
                safe_mem = text.replace("{", "{{").replace("}", "}}")
                safe_desc = desc.replace("{", "{{").replace("}", "}}")
                return prompt.format(memories=safe_mem, desc=safe_desc, memTok=memTok, maxTok=maxTok, percentage=percentage)

        # make memorie-token-count less then maxTok
        loop_count = 0
        while True:

            # break, if current tokens are less then maxTok
            cur_tok = self.memTok(table, tag, model)
            if cur_tok <= maxTok:
                logger.info(f"cleanTag [{table}/{tag}]: done — {cur_tok} tokens")
                break

            # update logger
            loop_count += 1
            logger.info(f"cleanTag [{table}/{tag}]: loop {loop_count} — {cur_tok} tokens (target: {maxTok})")

            # chop out oldest memories if taking too long
            if loop_count >= chopTime:
                logger.info(f"cleanTag [{table}/{tag}]: chopTime reached, deleting oldest memories from tag")

                rows = self.readMem(table, tag)  # list of mem, oldest first

                # find out how many old rows need to be deleted to get cur_tok to be less than maxTok.
                # the last row needed is cut down instead (keeping its newest text), 
                # so a tag that's one big row, like a summary, isn't wiped completely
                keysToDelete = []
                trimmed = None
                i = 0
                while cur_tok > maxTok and i < len(rows):
                    rowTok = self._strTok(rows[i][3])
                    if cur_tok - rowTok < maxTok:
                        keepChars = int(len(rows[i][3]) * (maxTok - (cur_tok - rowTok)) / rowTok)
                        if keepChars > 0:
                            trimmed = (rows[i][3][-keepChars:], rows[i][0])
                            break
                    cur_tok -= rowTok
                    keysToDelete.append(rows[i][0])
                    i += 1

                if keysToDelete or trimmed:
                    conn = sqlite3.connect(self.dbPath)
                    try:
                        cursor = conn.cursor()
                        batchSize = 500 # stay under SQLite's max bound parameter
                        for batchStart in range(0, len(keysToDelete), batchSize):
                            batch = keysToDelete[batchStart : batchStart + batchSize]
                            placeholders = ",".join("?" * len(batch))
                            cursor.execute(f"DELETE FROM {table} WHERE key IN ({placeholders})", batch)
                        if trimmed:
                            cursor.execute(f"UPDATE {table} SET memory = ? WHERE key = ?", trimmed)
                        conn.commit()
                    finally:
                        conn.close()

                logger.info(f"cleanTag [{table}/{tag}]: chop done — {self.memTok(table, tag, model)} tokens")
                break

            # if chopTime hasn't been reached
            # figure out how much to shrink by, and whether it fits in one prompt
            percentage = max(1, int((maxTok / cur_tok) * 100))
            text = self.rMem(table, tag)

            # token budget for memories = inputSize minus the prompt's own size
            budget = max(1, self.inputSize - self._strTok(_buildPrompt("", cur_tok, maxTok, percentage)))
            pieces = self.chunkText(text, model, budget)

            # if raw text bigger than the budget, summarize piece by piece
            if len(pieces) > 1:
                logger.info(f"cleanTag [{table}/{tag}]: {cur_tok} tokens > budget ({budget}) — splitting into {len(pieces)} chunks")
                summarizedPieces = []
                for piece in pieces:
                    pieceTok = self._strTok(piece)
                    pieceTarget = max(1, int(pieceTok * percentage / 100))
                    built_prompt = _buildPrompt(piece, pieceTok, pieceTarget, percentage)

                    # an empty summary would lose it's piece, keep it as it is, chopTime trims it later if needed
                    summarizedPieces.append(self._quickPrompt(built_prompt) or piece)

                # only touch the DB after every piece has summarized
                self.deleteTag(table, tag, 0)
                for summarizedPiece in summarizedPieces:
                    self.writeMem(table, tag, desc, summarizedPiece)

                continue  # go back and check cur_tok against maxTok

            # if whole tag fits in one prompt
            built_prompt = _buildPrompt(text, cur_tok, maxTok, percentage)
            shorterMem = self._quickPrompt(built_prompt)

            # an empty summary would lose it's piece, keep it as it is, chopTime trims it later if needed
            if not shorterMem:
                logger.warning(f"cleanTag [{table}/{tag}]: the model returned an empty summary - keeping the memories")
                continue
            self.deleteTag(table, tag, 0)
            self.writeMem(table, tag, desc, shorterMem)



    def tagMem(self, table:str, tag:str, memories:str, desc:str = None, prompt:str = None):

        # description = tag if none given
        if desc is None:
            desc = self.getDesc(table, tag)

        # memories = "" just makes the tag, with no memories in it
        if not memories.strip():
            self.writeMem(table, tag, desc, "")
            return

        # if no model in _init(), can't filter, keep the memories as is
        if self.model is None:
            logger.warning(f"tagMem [{table}/{tag}]: no model configured, skipping AI filtering")
            self.writeMem(table, tag, desc, memories)
            return


        def _buildPrompt(text):
            if prompt is None:
                return (f""" You are a memory filtering professional.
Here are all of the memories : {text}

Please filter the memories by the filter: "{desc}".
Record all the memories that are about, on the topic of, or including the filter : "{desc}"

It is typically better to let extra memories through, then to let not enough through.
Respond with only the filtered memories from a 1st person point of view (1st person from the assistant\'s POV).
If no memories match the filter, please respond with "Blank Entry"

New Filtered memories: """)
            else:
                safe_text = text.replace("{", "{{").replace("}", "}}")
                safe_desc = desc.replace("{", "{{").replace("}", "}}")
                return prompt.format(memories=safe_text, desc=safe_desc)

        entries = [m for m in memories.split("\n\n") if m.strip()]

        # token budget for memories = inputSize minus the prompt's own size
        overhead = self._strTok(_buildPrompt(""))
        budget = max(1, self.inputSize - overhead)

        # group memories into chunks that fit the budget
        chunks = []
        current_chunk = []
        current_tokens = 0
        for entry in entries:
            entry_tokens = self._strTok(entry)

            # split any entry that is bigger than the budget
            if entry_tokens > budget:
                pieces = [(piece, self._strTok(piece)) for piece in self.chunkText(entry, CW=budget)]
            else:
                pieces = [(entry, entry_tokens)]

            for piece, piece_tokens in pieces:
                # add 1 for the \n\n each piece is joined with
                if current_chunk and current_tokens + piece_tokens + 1 > budget:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [piece]
                    current_tokens = piece_tokens
                else:
                    current_chunk.append(piece)
                    current_tokens += piece_tokens + 1
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        # filter every chunk, dropping empty / Blank Entry replies
        results = [self._quickPrompt(_buildPrompt(chunk)) for chunk in chunks]

        # only write once every chunk has been filtered, so a failure partway writes nothing
        for result in results:
            if result.strip() and result.strip().lower() != "blank entry":
                self.writeMem(table, tag, desc, result)



    def getTags(self, table:str, drop:list = None) -> list:
        self._validateTable(table)

        conn = sqlite3.connect(self.dbPath)
        try:
            cursor = conn.cursor()

            # if the table hasn't been created yet, there are no tags to return
            if not self._tableExists(cursor, table):
                return []

            # set tags to a list of every unique tag in the table.
            cursor.execute(f"SELECT DISTINCT tag FROM {table}")
            rows = cursor.fetchall()
            tags = [row[0] for row in rows]
        finally:
            conn.close()

        if drop:
            tags = [t for t in tags if t not in drop]

        return tags



    # to edit an existing tag's description
    def setTagDesc(self, table:str, tag:str, desc:str):
        self._validateTable(table)

        conn = sqlite3.connect(self.dbPath)
        try:
            cursor = conn.cursor()

            # if the table hasn't been created yet, there are no tags to update
            if not self._tableExists(cursor, table):
                return

            # update desc based on tag
            cursor.execute(f"UPDATE {table} SET desc = ? WHERE tag = ?", (desc, tag))

            conn.commit()
        finally:
            conn.close()



    def getDesc(self, table:str, tag:str) -> str:
        # pull the latest memory with given tag
        memory = self.readMem(table, tag, 1)

        #return that memory's description
        return memory[0][2] if memory else tag
    





    # < - - - - - [ Utility ] - - - - - >

    # I just think the functtion name is easier to remember
    def memRows(self, table:str, tag:str = "all") -> int:
        return len(self.readMem(table, tag, 0))
    


    def memTok(self, table:str, tag:str = "all", model:str = None) -> int:
        return self._strTok(self.rMem(table, tag))



    def chunkText(self, text:str, model:str = None, CW:int = None) -> list:
        if CW is None:
            CW = self.inputSize
        if model is None:
            model = self.model

        textTokens = self._strTok(text)

        # if it already fits, nothing to do
        if textTokens <= CW:
            return [text]

        # highball chars-per-token as a safety margin
        overEstimate = 1.2
        charsPerToken = len(text) / (textTokens * overEstimate)
        chunkChars = max(1, int(CW * charsPerToken))  # never 0, so loop always ends.

        # Chop up the string into a list
        chunks = []
        charStart = 0
        while charStart < len(text):
            chunks.append(text[charStart : charStart + chunkChars])
            charStart += chunkChars

        return chunks



    def _strTok(self, text:str) -> int:

        # exact count, if a tokenizerModel was given
        if self.tokenizerModel is not None:
            return len(self.tokenizerModel.encode(text))

        # otherwise estimate
        cjk = len(_CJK.findall(text))
        return cjk + math.ceil((len(text) - cjk) / CHARS_PER_TOKEN)




    @staticmethod
    def now() -> str:
        dt = datetime.now()
        date = dt.strftime("%Y-%m-%d")
        time = dt.strftime("%H:%M:%S")
        return f"{date} // {time}"
    


    # claude wrote this function, it helps with security, 
    # and ensureing the table name given doesn't throw an error.
    @staticmethod
    def _validateTable(table:str):
        # only letters/digits/underscores, max 256 chars
        if not isinstance(table, str) or not re.fullmatch(r'[A-Za-z0-9_]{1,256}', table):
            logger.error(f"Invalid table name: '{table}'")
            raise ValueError(f"Invalid table name: '{table}'")

        # some names pass the check above but SQLite still rejects them:
        # starting with a digit ("123"), SQL keywords ("order"), or "sqlite_..."
        # so let SQLite try the name on a throwaway in-memory database.
        # safe to put in the SQL, the check above only allows letters/digits/underscores
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(f"CREATE TABLE {table} (x)")
            conn.execute(f"SELECT x FROM {table}")
        except sqlite3.Error as e:
            logger.error(f"Invalid table name: '{table}' — SQLite rejects it ({e})")
            raise ValueError(f"Invalid table name: '{table}' — SQLite rejects it ({e})") from None
        finally:
            conn.close()



    @staticmethod
    def _tableExists(cursor, table:str) -> bool:
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        return cursor.fetchone() is not None






# < - - - - - [ AI ] - - - - - >

    # Takes ollama.chat style messages. Returns the raw ollama response
    def useAI(self, messages:list, tools:list = None, model:str = None):
        # use default model if not specified
        if model is None:
            model=self.model
            # if default is none return error
            if model is None:
                logger.error("LivingMemory.useAI() requires a model. Example: LivingMemory.useAI(messages='message', model='qwen3:14b') ")
                raise ValueError("LivingMemory.useAI() requires a model. Example: LivingMemory.useAI(messages='message', model='qwen3:14b') ")
        
        # try to run AI
        try:
            return self._client.chat(
                model=model,
                messages=messages,
                tools=tools,
                # num_predict caps the reply at outputSize, so it can't be too big for Ollama
                options={"num_ctx": self.contextWindow, "num_predict": self.outputSize},
            )

        # these were all written by claude to help solve an error massage, safe to remove if desired
        except ollama.ResponseError as e:
            logger.error(f"Ollama error: {e.error}")
            raise RuntimeError(f"[Ollama error: {e.error}]") from e
        except ConnectionError as e:
            logger.error(f"Couldn't reach Ollama: {e}")
            raise RuntimeError(f"[Error: couldn't reach Ollama {e}]") from e
        except Exception as e:
            logger.error(f"Unexpected error in useAI: {e}")
            raise RuntimeError(f"[Error: {e}]") from e



    # used to send useAI() a single prompt with no history
    def _quickPrompt(self, prompt: str, model:str = None) -> str:
        response = self.useAI([{"role": "user", "content": prompt}], model=model)
        return response.message.content.strip()






# < - - - - - [ Core LivingMemory ] - - - - - >

    # with memory = "", the row just makes the tag exist.
    def writeMem(self, table:str, tag:str = "all", desc:str = "all", memory:str = "", time:str = None):
        self._validateTable(table)

        # if user didn't set description, use tag as description
        if desc == "all":
            desc = tag

        # if user didn't set time, match it with the time
        if time is None:
            time = self.now()

        conn = sqlite3.connect(self.dbPath)
        try:
            cursor = conn.cursor()

            # create table if it doesn't exist
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    key          INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag          TEXT,
                    desc         TEXT,
                    memory       TEXT,
                    time         TEXT NOT NULL
                )
            """)

            # write the new memory
            cursor.execute(f"""
                INSERT INTO {table} (tag, desc, memory, time)
                VALUES (?, ?, ?, ?)
            """, (tag, desc, memory, time))
            conn.commit()
        finally:
            conn.close()



    # readMem() but returns a single, big, string
    def rMem(self, table:str, tag:str = "all", latest:int = 0, time:bool = False) -> str:
        memories = [m for m in self.readMem(table, tag, latest) if m[3].strip()]
        if time:
            return "\n\n".join(f"{memory[4]}\n{memory[3]}" for memory in memories)
        return "\n\n".join(memory[3] for memory in memories)
    


    def readMem(self, table:str, tag:str = "all", latest:int = 0):
        self._validateTable(table)

        conn = sqlite3.connect(self.dbPath)
        try:
            cursor = conn.cursor()

            # if the table hasn't been created yet, there are no rows to return
            if not self._tableExists(cursor, table):
                return []

            # tag = "all" means all tags and latest = 0 means all rows
            if tag != "all":
                if latest > 0:
                    cursor.execute(f"""
                        SELECT * FROM (
                            SELECT * FROM {table} WHERE tag = ?
                            ORDER BY time DESC, key DESC LIMIT ?
                        ) ORDER BY time ASC, key ASC
                    """, (tag, latest))
                else:
                    cursor.execute(
                        f"SELECT * FROM {table} WHERE tag = ? ORDER BY time ASC, key ASC",
                        (tag,),
                    )
            else:
                if latest > 0:
                    cursor.execute(f"""
                        SELECT * FROM (
                            SELECT * FROM {table}
                            ORDER BY time DESC, key DESC LIMIT ?
                        ) ORDER BY time ASC, key ASC
                    """, (latest,))
                else:
                    cursor.execute(f"SELECT * FROM {table} ORDER BY time ASC, key ASC")

            rows = cursor.fetchall()

            # return each memory with tag, and it's meta data
            return rows
        finally:
            conn.close()



    def deleteTag(self, table:str, tag:str, keep:int = 0):
        self._validateTable(table)

        conn = sqlite3.connect(self.dbPath)
        try:
            cursor = conn.cursor()

            # nothing to delete if the table doesn't exist yet
            if not self._tableExists(cursor, table):
                return

            if keep > 0:
                # delete all rows of tag except the newest "keep"
                cursor.execute(f"""
                    DELETE FROM {table} WHERE tag = ?
                    AND key NOT IN (
                        SELECT key FROM {table} WHERE tag = ?
                        ORDER BY time DESC, key DESC LIMIT ?
                    )
                """, (tag, tag, keep))
            else:
                # delete all rows of tag
                cursor.execute(f"DELETE FROM {table} WHERE tag = ?", (tag,))

            conn.commit()
        finally:
            conn.close()

